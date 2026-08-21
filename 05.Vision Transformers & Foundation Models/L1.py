#Weighted Sum
import torch
features = torch.tensor([
    [2., 1.],
    [0., 3.],
    [4., 2.]
])

weights = torch.tensor([0.2, 0.3, 0.5])

output = (weights.unsqueeze(1) * features).sum(dim=0)

print(output)

#Similarity Scores

Q = torch.tensor([[1., 0.]])
K = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 1.]
])

scores = Q @ K.T

print(scores)

#A Tiny Transformer Encoder

import torch
import torch.nn as nn
encoder = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True
)
x = torch.randn(2, 16, 128)
output = encoder(x)
print(output.shape)