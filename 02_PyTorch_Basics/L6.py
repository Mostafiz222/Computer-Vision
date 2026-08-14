import torch
import torch.nn as nn

# Create a linear layer with 4 input features and 3 output neurons
layer = nn.Linear(4, 3)

# Create a random input tensor with a batch size of 2
x = torch.randn(2, 4)
#imagin 2 image,each image has 4 feature(input)

# Forward pass
y = layer(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
print(layer.weight)
print(layer.bias)