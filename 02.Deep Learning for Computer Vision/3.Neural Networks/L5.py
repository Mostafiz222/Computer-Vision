import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Simple model
model = nn.Sequential(
    nn.Linear(2, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# One sample
x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[5.0]])

print("Weight BEFORE training:")
print(model[0].weight)
print(model[0].bias)

# Forward pass
y_pred = model(x)

# Compute loss
loss = criterion(y_pred, y_true)

# Compute gradients
loss.backward()

print("\nGradient:")
print(model[0].weight.grad)
print(model[0].bias.grad)

# Update parameters
optimizer.step()

print("\nWeight AFTER optimizer.step():")
print(model[0].weight)
print(model[0].bias)