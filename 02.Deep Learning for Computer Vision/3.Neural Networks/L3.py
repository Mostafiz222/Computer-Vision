#nn.sequential
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

print(model)
x = torch.randn(5, 4)
y = model(x)

print("\nInput shape :", x.shape)
print("Output shape:", y.shape)

print("\nModel Parameters:")
for param in model.parameters():
    print(param.shape)

#activation function:
import torch
import torch.nn as nn

# Model without activation
model_linear = nn.Sequential(
    nn.Linear(4, 8),
    nn.Linear(8, 2)
)

# Model with activation
model_relu = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

print("Without Activation:\n")
print(model_linear)

print("\nWith ReLU:\n")
print(model_relu)

#ReLU (Rectified Linear Unit).

import torch
import torch.nn as nn
relu = nn.ReLU()
x = torch.tensor([-5., -2., -1., 0., 1., 3., 7.])
y = relu(x)
print("Input :", x)
print("Output:", y)

x = torch.randn(10)
print("Random Input:")
print(x)
print("\nAfter ReLU:")
print(relu(x))