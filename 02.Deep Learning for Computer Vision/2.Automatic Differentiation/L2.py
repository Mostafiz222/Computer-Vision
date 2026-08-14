import torch
import torch.nn as nn

# 4. Implement SimpleNetwork class
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate model
model = SimpleNetwork()

# 5. Create random input tensor of shape (5, 3) and verify output shape
x_input = torch.rand(5, 3)
output = model(x_input)

print("--- Output Shape Verification ---")
print("Input Shape: ", x_input.shape)
print("Output Shape:", output.shape)
assert output.shape == (5, 2), "Output shape does not match expected (5, 2)"

print("\n--- Model Learnable Parameters ---")
# 6. Print names and shapes of all learnable parameters
for name, param in model.named_parameters():
    print(f"{name:12s} | Shape: {list(param.shape)}")