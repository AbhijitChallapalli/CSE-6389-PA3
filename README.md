# AD vs CN Classification from fMRI Time Series  
*CSE 6389 – Advanced Medical Imaging & Graphs – Programming Assignment 3*

This project implements and compares two recurrent neural network approaches for **Alzheimer’s Disease (AD) vs Cognitively Normal (CN)** classification from resting-state fMRI time series:

1. **MinimalRNN (Nguyen-style)**
2. **Stacked LSTM (Li-style) with sliding-window data augmentation**

Training and evaluation are done at the **subject level** using **5-fold stratified cross-validation**. For each fold, the code reports validation metrics and saves **confusion matrix heatmaps**.

---

## 1. Dataset & Directory Structure

The code assumes the following dataset organization under `DATA_ROOT = "./"`:

```text
.
├── AD1/
│   └── fmri_average_signal/
│       ├── raw_fmri_feature_matrix_1.txt
│       ├── raw_fmri_feature_matrix_2.txt
│       └── ...
├── AD2/
│   └── fmri_average_signal/
│       └── raw_fmri_feature_matrix_*.txt
├── ...
├── AD10/
├── CN1/
│   └── fmri_average_signal/
│       └── raw_fmri_feature_matrix_*.txt
├── ...
└── CN10/
```

- Each subject folder is named either **`AD*`** or **`CN*`**.
- Inside each subject:
  - `fmri_average_signal/` contains multiple time-point text files:
    - `raw_fmri_feature_matrix_1.txt`
    - `raw_fmri_feature_matrix_2.txt`
    - …
- Each `raw_fmri_feature_matrix_*.txt` file stores a **vector of ROI features** for one time point
  (flattened to shape `(R,)`, where `R = 150` ROIs in this assignment).

The loader:

- Reads all `raw_fmri_feature_matrix_*.txt` files,
- Sorts them by the integer in the filename (1,2,…),
- Stacks them into a subject matrix **`X_subj` of shape `(T, R)`**,
- Where `T = 101` time points and `R = 150` ROIs in the given data.

---

## 2. Preprocessing

### 2.1 Per-Subject z-Scoring (Temporal Normalization)

For each subject, the data `X_subj ∈ ℝ^{T×R}` is normalized **independently**:

```python
mean = X_subj.mean(axis=0, keepdims=True)  # (1, R)
std  = X_subj.std(axis=0, keepdims=True)   # (1, R)
X_z  = (X_subj - mean) / (std + 1e-6)
```

- Normalization is done **per subject and per ROI** across time.
- Controlled by `USE_ZSCORE = True`.

This helps stabilize training and prevents ROIs with large absolute values from dominating the loss.

### 2.2 Sliding-Window Data Augmentation (Li-Style)

For training and validation, we use a **sliding window** approach:

- **Window size**: `WINDOW_SIZE = 40` time points  
- **Stride**: `WINDOW_STRIDE = 20` time points  

For each subject `s` with time series length `T`:

- We generate windows:
  - `[0:40]`, `[20:60]`, `[40:80]`, …  
- Each window has shape `(L, R)` with `L = 40`.
- Each window inherits the subject’s binary label (AD=1, CN=0).

This is implemented in `SlidingWindowDataset`, which returns:

```python
seq_tensor   # (L, R)  windowed fMRI sequence
label_tensor # scalar  subject label (0 or 1)
subj_idx     # index of the subject this window belongs to
```

---

## 3. Models

### 3.1 MinimalRNN (Nguyen-Style)

The MinimalRNN model is implemented in `MinimalRNN_AD_Classifier`:

- Input: `(B, T, input_dim)` where:
  - `B`: batch size
  - `T`: sequence length
  - `input_dim = R = 150` (ROIs)
- Hidden state dimension: `HIDDEN_DIM = 64`

Update equations:

- Let `x_t ∈ ℝ^{input_dim}` be the input at time `t`
- `z_t = tanh(W_x x_t + b_x)`
- `u_t = σ(W_h h_{t−1} + V z_t + b_u)`
- `h_t = u_t * h_{t−1} + (1 − u_t) * z_t`

After processing the full sequence:

- Use the final hidden state `h_T` as sequence representation.
- Pass through an MLP:
  - `h_T → ReLU(Linear(64 → 16)) → Linear(16 → 1)` → **logit** for AD vs CN.

### 3.2 LSTM (Li-Style)

The Li-style LSTM is implemented in `LSTM_AD_Classifier`:

- Two stacked LSTMs:
  - LSTM1: `input_dim = 150`, `hidden_size = 64`
  - LSTM2: `input_dim = 64`, `hidden_size = 32`
- Use the last time step of the second LSTM (`h_T`) as representation.
- MLP head:
  - `h_T → ReLU(Linear(32 → 16)) → Linear(16 → 1)` → **logit**.

Both models output raw logits which are passed to `BCEWithLogitsLoss` during training and to a `sigmoid` at evaluation time.

---

## 4. Training & Evaluation Protocol

### 4.1 Cross-Validation Setup

We use **5-fold stratified cross-validation** at the **subject level**:

- `NUM_FOLDS = 5`
- Split is done using `StratifiedKFold` on the **20 subjects** (10 AD, 10 CN).
- Each fold:
  - ~16 subjects for training
  - ~4 subjects for validation

This ensures that **no subject leaks between train and validation**.

### 4.2 Training Hyperparameters

- Number of epochs: `NUM_EPOCHS = 50`
- Batch size: `BATCH_SIZE = 8`
- Optimizer: `Adam` with `lr = 1e-3`
- Loss: `BCEWithLogitsLoss`
- Device: GPU (`cuda`) if available, else CPU.

Randomness is controlled with:

```python
SEED = 42
set_seed(SEED)
```

This sets seeds for `random`, `numpy`, and `torch`, and enables deterministic settings for cuDNN.

### 4.3 Subject-Level Evaluation (Aggregation over Windows)

Evaluation is done at the **subject level** by aggregating all windows belonging to the same subject:

1. For each window, the model outputs a logit `ℓ`.
2. For each subject `s`, collect all logits `{ℓ₁, ℓ₂, …}` from its windows.
3. Compute the **mean logit**:  
   `mean_logit_s = mean({ℓᵢ})`
4. Convert to probability with `sigmoid`:
   `p_s = σ(mean_logit_s)`
5. Threshold at 0.5:
   - `p_s ≥ 0.5 → AD (1)`
   - `p_s < 0.5 → CN (0)`

Then we compute:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix (2 × 2: True CN / AD vs Predicted CN / AD)

using `sklearn.metrics`.

### 4.4 Model Selection per Fold

For each fold, we track the **best epoch based on F1 score**:

```python
if metrics["f1"] > best_f1:
    best_f1 = metrics["f1"]
    best_metrics = metrics
```

At the end of the fold, we store `best_metrics` as that fold’s result and print:

- Best F1
- Confusion matrix

This ensures each fold’s reported metrics correspond to the epoch where the model performs best on validation F1.

---

## 5. Confusion Matrix Visualization

For every fold, a **confusion matrix heatmap** is generated and saved:

```python
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
```

- Files are saved as:
  - `cm_minimalrnn_fold_1.png`, …, `cm_minimalrnn_fold_5.png`
  - `cm_lstm_fold_1.png`, …, `cm_lstm_fold_5.png`
- These are useful for qualitative analysis of true positives, false positives, etc.

---

## 6. Results & Ablation Study

On the provided dataset (20 subjects: 10 AD, 10 CN), a sample run produced:

### 6.1 MinimalRNN (Nguyen-Style)

```text
=== Cross-validated results ===
Accuracy : 0.600 ± 0.122
Precision: 0.633 ± 0.194
Recall   : 0.800 ± 0.245
F1       : 0.660 ± 0.095
```

### 6.2 LSTM (Li-Style)

```text
=== Cross-validated results ===
Accuracy : 0.900 ± 0.200
Precision: 0.900 ± 0.200
Recall   : 1.000 ± 0.000
F1       : 0.933 ± 0.133
```

> **Note:**  
> Due to the **very small sample size (N = 20 subjects)**, validation performance can vary between runs, and overfitting is possible. The goal of the assignment is primarily to:
> - Implement the architectures correctly,
> - Use subject-level cross-validation,
> - Compare behavior of MinimalRNN vs LSTM on fMRI time-series.

### 6.3 Ablation Study (Model-Level Comparison)

The table below summarizes the main ablation across model architectures while keeping the dataset, preprocessing, windowing, and training hyperparameters fixed.

| Variant ID | Model Type  | Sliding-Window (40 / 20) | Hidden Dim(s)     | Accuracy (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) | Notes                                |
|-----------|-------------|--------------------------|-------------------|------------------------|-------------------------|----------------------|------------------|--------------------------------------|
| A         | MinimalRNN  | Yes                      | 64                | 0.600 ± 0.122          | 0.633 ± 0.194           | 0.800 ± 0.245        | 0.660 ± 0.095    | Simpler recurrent cell, fewer gates |
| B         | LSTM (Li)   | Yes                      | 64 → 32 (2-layer) | 0.900 ± 0.200          | 0.900 ± 0.200           | 1.000 ± 0.000        | 0.933 ± 0.133    | Higher capacity, stacked LSTMs      |

**Observations (for report):**

- Moving from MinimalRNN (Variant A) to a stacked LSTM (Variant B) substantially increases accuracy and F1 on this small dataset.
- Both models use the same **sliding-window augmentation** and **subject-level aggregation**, so the performance gain is attributable mainly to:
  - The greater expressivity of the LSTM cell,
  - The additional depth (two LSTM layers),
  - Better ability to capture complex temporal dynamics in the fMRI time series.
- However, because the dataset is tiny, the high scores for the LSTM may partly reflect **overfitting**; this should be discussed in the assignment write-up.

---

## 7. Code Structure

Main file:

- **`train.py`**  
  - Data loading & z-scoring (`load_all_subjects`, `zscore_subject`)
  - Sliding-window dataset (`SlidingWindowDataset`)
  - Models (`MinimalRNN_AD_Classifier`, `LSTM_AD_Classifier`)
  - Training utility (`train_one_epoch`)
  - Subject-level evaluation (`evaluate_subject_level`)
  - K-fold orchestration (`run_kfold`)
  - Main script:
    - Loads data
    - Runs:
      - MinimalRNN cross-validation
      - LSTM cross-validation

Output:

- Console logs for each fold and epoch.
- Confusion matrices per fold saved as `.png`.

---

## 8. Dependencies

Recommended environment:

- Python 3.8+
- Packages:
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

You can install requirements via:

```bash
pip install numpy torch scikit-learn matplotlib seaborn
```

---

## 9. How to Run

1. Place the data in the directory structure described in Section 1.
2. Ensure you are in the project root (where `train.py` is located).
3. Run:

```bash
python train.py
```

The script will:

1. Load and preprocess all 20 subjects.
2. Print a list of subjects and labels (AD=1, CN=0).
3. Run 5-fold CV for:
   - MinimalRNN model
   - LSTM model
4. Print fold-wise and cross-validated metrics.
5. Save confusion matrix images for each fold and model.

---

## 10. Notes & Possible Extensions

- **Reproducibility:**  
  The seed is fixed (`SEED = 42`) for `numpy`, `random`, and `torch` to help reproduce results.

- **Hyperparameter Tuning:**  
  You can experiment with:
  - Hidden dimensions (`HIDDEN_DIM`)
  - Window size / stride (`WINDOW_SIZE`, `WINDOW_STRIDE`)
  - Number of layers in LSTM
  - Learning rate, batch size

- **Additional Metrics:**  
  For the assignment report, you may also want:
  - ROC curves / AUC
  - Per-fold subject-level prediction tables
  - Discussion of failure modes seen in confusion matrices.

---

## 11. Summary (for Assignment Write-Up)

This project demonstrates:

- How to convert raw fMRI time-series (per-time-point ROI vectors) into an input suitable for RNNs.
- How to implement and train two different recurrent architectures for AD vs CN classification:
  - MinimalRNN (Nguyen-style)
  - Stacked LSTM (Li-style)
- How to apply **sliding-window augmentation** to increase the effective number of training sequences.
- How to perform **subject-level cross-validated evaluation** and visualize confusion matrices.
- How simple architectural ablations (MinimalRNN vs stacked LSTM) impact performance on the same fMRI dataset.

The provided implementation and results can be directly used as the basis for the assignment report, with added interpretation and comparison of MinimalRNN vs LSTM behavior on this small fMRI dataset.
