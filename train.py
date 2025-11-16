import os
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Root directory where AD1, AD2, ..., CN1, ... folders live
DATA_ROOT = "./"

# Folder name 
FMRI_FOLDER_NAME = "fmri_average_signal"

# raw feature txt files
TIME_FILE_GLOB = "raw_fmri_feature_matrix_*.txt"

# Sliding-window augmentation parameters (Li et al. used 40 / 20)
WINDOW_SIZE = 40
WINDOW_STRIDE = 20

NUM_FOLDS = 5
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64  # LSTM/MinimalRNN hidden size

# per-subject z-scoring across time
USE_ZSCORE = True  
SEED = 42


# Seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Data loading & preprocessing
def zscore_subject(X_subj, eps=1e-6):
    """
    Per-subject z-scoring across time for each ROI.

    X_subj: numpy array of shape (T, R)
    Returns: (T, R) normalized
    """
    mean = X_subj.mean(axis=0, keepdims=True)  # (1, R)
    std = X_subj.std(axis=0, keepdims=True)    # (1, R)
    return (X_subj - mean) / (std + eps)


def extract_int_from_raw_txt(stem):
    """
    Helper to sort timepoint files by the integer in the filename.
    E.g., "raw_fmri_feature_matrix_10" -> 10
    """
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def load_subject(subject_dir,fmri_folder_name=FMRI_FOLDER_NAME,time_file_glob=TIME_FILE_GLOB):
    """
    Load one subject's fMRI time series as a (T, R) numpy array.

    subject_dir: Path to e.g. .../AD1
    Returns: X_subj, shape (T, R)
    """
    subject_dir = Path(subject_dir)
    signal_dir = subject_dir / fmri_folder_name

    if not signal_dir.exists():
        raise FileNotFoundError(
            f"fmri folder '{fmri_folder_name}' not found in {subject_dir}"
        )

    time_files = sorted(
        signal_dir.glob(time_file_glob),
        key=lambda p: extract_int_from_raw_txt(p.stem),
    )

    if len(time_files) == 0:
        raise RuntimeError(f"No timepoint files found in {signal_dir}")

    time_series = []
    for f in time_files:
        arr = np.loadtxt(f)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)  # flatten to (R,)

        # # Optional sanity check: verify ROI count
        # if EXPECTED_NUM_ROIS is not None and arr.size != EXPECTED_NUM_ROIS:
        #     raise ValueError(
        #         f"File {f} has {arr.size} values, expected {EXPECTED_NUM_ROIS}."
        #     )

        time_series.append(arr)

    X_subj = np.stack(time_series, axis=0)  # (T, R)
    return X_subj


def load_all_subjects(data_root,fmri_folder_name=FMRI_FOLDER_NAME,time_file_glob=TIME_FILE_GLOB,zscore=USE_ZSCORE):
    """
    Load all AD* and CN* subject folders under data_root.

    Returns:
        X: (N, T, R) numpy array
        y: (N,) numpy array, AD=1, CN=0
        subject_dirs: list of (Path, label) in the same order as X/y
    """
    data_root = Path(data_root)
    #print(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    subject_dirs = []
    labels = []

    # AD subjects: label 1
    for subj_dir in sorted(data_root.glob("AD*")):
        if subj_dir.is_dir():
            subject_dirs.append((subj_dir, 1))
    # CN subjects: label 0
    for subj_dir in sorted(data_root.glob("CN*")):
        if subj_dir.is_dir():
            subject_dirs.append((subj_dir, 0))

    if len(subject_dirs) == 0:
        raise RuntimeError("No AD* or CN* subject folders found under DATA_ROOT.")

    all_subject_data = []
    for subj_dir, label in subject_dirs:
        X_subj = load_subject(subj_dir, fmri_folder_name, time_file_glob)
        if zscore:
            X_subj = zscore_subject(X_subj)
        all_subject_data.append(X_subj)
        labels.append(label)

    X = np.stack(all_subject_data, axis=0)  # (N, T, R)
    y = np.array(labels, dtype=np.int64)    # (N,)
    return X, y, subject_dirs


#Li-style sliding windows
class SlidingWindowDataset(Dataset):
    """
    Builds overlapping subsequences (windows) from subject-level time series.

    This is the Li-style data augmentation:
      - Window size L (e.g., 40)
      - Stride S (e.g., 20)
    For each subject, we generate windows:
      [0:L], [S:S+L], [2S:2S+L], ...

    Each window gets the subject's label.
    For evaluation, we aggregate window-level predictions back to subject-level.
    """

    def __init__(self, X, y, subject_indices,
                 window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
        """
        X: numpy array (N, T, R)
        y: numpy array (N,)
        subject_indices: list/array of subject indices to include in this dataset
        """
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride

        self.samples = []  # list of (subj_idx, start_idx)

        for subj_idx in subject_indices:
            T = X[subj_idx].shape[0]
            starts = list(range(0, T - window_size + 1, stride))
            if len(starts) == 0:
                # Fallback: one window starting at 0 if something is off
                starts = [0]
            for s in starts:
                self.samples.append((int(subj_idx), s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_idx, start = self.samples[idx]
        X_subj = self.X[subj_idx]  # (T, R)
        seq = X_subj[start:start + self.window_size, :]  # (L, R)
        label = self.y[subj_idx]   # scalar

        seq_tensor = torch.from_numpy(seq).float()               # (L, R)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        subj_idx_tensor = torch.tensor(subj_idx, dtype=torch.long)

        return seq_tensor, label_tensor, subj_idx_tensor


# Models
class LSTM_AD_Classifier(nn.Module):
    """
    Li-style stacked LSTM, but shrunk for small sample size.

    Input: (B, T, R)
    - LSTM1: hidden_size = hidden1
    - LSTM2: hidden_size = hidden2
    - Use last time step of LSTM2 as sequence representation.
    - MLP head -> 1 logit (AD vs CN)
    """

    def __init__(self, input_dim=150, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden2, 16)
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        """
        x: (B, T, R)
        """
        out1, _ = self.lstm1(x)         # (B, T, hidden1)
        out2, _ = self.lstm2(out1)      # (B, T, hidden2)
        h_T = out2[:, -1, :]            # (B, hidden2)
        h = torch.relu(self.fc1(h_T))   # (B, 16)
        logits = self.fc_out(h).squeeze(-1)  # (B,)
        return logits


class MinimalRNN_AD_Classifier(nn.Module):
    """
    Nguyen-style MinimalRNN for AD vs CN.

    Update:
      z_t = tanh(W_x x_t + b_x)
      u_t = sigmoid(W_h h_{t-1} + V z_t + b_u)
      h_t = u_t * h_{t-1} + (1 - u_t) * z_t

    Then use h_T -> MLP -> 1 logit.
    """

    def __init__(self, input_dim=150, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.gate_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.fc1 = nn.Linear(hidden_dim, 16)
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.size()
        device = x.device

        # Initialize h_0 = 0
        h = torch.zeros(B, self.hidden_dim, device=device)

        for t in range(T):
            x_t = x[:, t, :]  # (B, input_dim)
            z_t = torch.tanh(self.input_to_hidden(x_t))  # (B, hidden_dim)
            u_t = torch.sigmoid(self.gate_h(h) + self.gate_z(z_t))  # (B, hidden_dim)
            h = u_t * h + (1.0 - u_t) * z_t

        h = torch.relu(self.fc1(h))
        logits = self.fc_out(h).squeeze(-1)  # (B,)
        return logits


def create_model(model_type, input_dim, hidden_dim=HIDDEN_DIM):
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTM_AD_Classifier(input_dim=input_dim,hidden1=hidden_dim,hidden2=max(16, hidden_dim // 2))
    elif model_type in ("minimalrnn", "minimal"):
        return MinimalRNN_AD_Classifier(input_dim=input_dim,hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Training
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, labels, subj_idx in train_loader:
        sequences = sequences.to(device)  # (B, L, R)
        labels = labels.to(device)       # (B,)

        optimizer.zero_grad()
        logits = model(sequences)        # (B,)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = sequences.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_subject_level(model, val_loader, device, num_subjects):
    """
    Aggregate window-level predictions back to subject-level.

    For each subject:
      - collect all window logits
      - average logits -> prob = sigmoid(mean_logit)
      - threshold at 0.5 to get AD vs CN

    Returns dict with metrics and per-subject predictions.
    """
    model.eval()
    logits_per_subject = {i: [] for i in range(num_subjects)}
    labels_per_subject = {}

    with torch.no_grad():
        for sequences, labels, subj_idx in val_loader:
            sequences = sequences.to(device)
            logits = model(sequences).cpu().numpy()  # (B,)
            subj_idx = subj_idx.cpu().numpy()        # (B,)
            labels = labels.cpu().numpy()            # (B,)

            for logit, s_idx, lab in zip(logits, subj_idx, labels):
                logits_per_subject[int(s_idx)].append(float(logit))
                labels_per_subject[int(s_idx)] = int(lab)

    y_true = []
    y_pred = []

    for s_idx, logit_list in logits_per_subject.items():
        if s_idx not in labels_per_subject:
            continue
        mean_logit = float(np.mean(logit_list))
        prob = 1.0 / (1.0 + np.exp(-mean_logit))
        pred = 1 if prob >= 0.5 else 0

        y_pred.append(pred)
        y_true.append(labels_per_subject[s_idx])

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_kfold(X, y, model_type="lstm",
              window_size=WINDOW_SIZE,
              stride=WINDOW_STRIDE,
              num_folds=NUM_FOLDS,
              num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              lr=LEARNING_RATE,
              hidden_dim=HIDDEN_DIM):
    """
    Run stratified K-fold CV at subject level.

    For each fold:
      - build SlidingWindowDataset for train and val
      - train model
      - evaluate subject-level metrics (aggregated over windows)
      - keep best epoch's metrics by F1-score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N, T, R = X.shape
    print(f"Data shape: N={N} subjects, T={T} time points, R={R} ROIs")

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold_idx}/{num_folds} ===")
        print(f"Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")

        train_dataset = SlidingWindowDataset(
            X, y, train_idx, window_size=window_size, stride=stride
        )
        val_dataset = SlidingWindowDataset(
            X, y, val_idx, window_size=window_size, stride=stride
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        model = create_model(model_type, input_dim=R, hidden_dim=hidden_dim)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_f1 = 0.0
        best_metrics = None

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            metrics = evaluate_subject_level(
                model, val_loader, device, num_subjects=N
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = metrics

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.3f} | "
                f"Val F1: {metrics['f1']:.3f}"
            )

        print(f"Best F1 for fold {fold_idx}: {best_f1:.3f}")
        print("Confusion matrix:\n", best_metrics["confusion_matrix"])
        fold_results.append(best_metrics)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            best_metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="mako",     
            cbar=False,
            ax=ax,
            xticklabels=["CN", "AD"],
            yticklabels=["CN", "AD"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — Fold {fold_idx}")
        plt.tight_layout()
        plt.savefig(f"cm_{model_type}_fold_{fold_idx}.png", dpi=200)
        plt.close(fig) 

    accs = [res["accuracy"] for res in fold_results]
    precs = [res["precision"] for res in fold_results]
    recs = [res["recall"] for res in fold_results]
    f1s = [res["f1"] for res in fold_results]

    print("\n=== Cross-validated results ===")
    print(f"Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
    print(f"Recall   : {np.mean(recs):.3f} ± {np.std(recs):.3f}")
    print(f"F1       : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


if __name__ == "__main__":
    # Load data as (N, T, R)
    X, y, subject_dirs = load_all_subjects(DATA_ROOT)
    print("Subjects loaded:")
    for i, (p, label) in enumerate(subject_dirs):
        print(f"  #{i:02d}: {p.name}, label={label}")

    # Run MinimalRNN (Nguyen-style) as an ablation
    print("\nRunning MinimalRNN model (Nguyen-style)...")
    run_kfold(X, y, model_type="minimalrnn")
    
    # Run Li-style LSTM with sliding-window augmentation
    print("\nRunning LSTM model (Li-style)...")
    run_kfold(X, y, model_type="lstm")


