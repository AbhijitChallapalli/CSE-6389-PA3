#More details about RNN at https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
import torch
import torch.nn as nn

rnn = nn.RNN(10, 20, 2)   # (input_size, hidden_size, num_layers)
inputs = torch.randn(5, 3, 10)  # (time_step, batch_size, input_size)
h0 = torch.randn(2, 3, 20)  # (num_layers, batch_size, hidden_size)
output, hn = rnn(inputs, h0)
print(output.shape)  # (time_step, batch_size, hidden_size)

for name, param in rnn.named_parameters():
    if param.requires_grad:
        print(name, param.size())

