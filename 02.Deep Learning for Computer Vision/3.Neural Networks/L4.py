#Forward Pass
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

x = torch.randn(3, 4)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)

print("\nOutput:")
print(output)

#Loss
import torch
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
# Logits for 3 samples, 4 classes
outputs = torch.tensor([
    [2.5, 0.3, 1.1, -0.2],
    [0.1, 2.2, -0.4, 1.5],
    [1.0, 0.2, 3.0, -1.0]
])

# Correct class indices
labels = torch.tensor([0, 1, 2])

loss = criterion(outputs, labels)

print("Loss:", loss.item())