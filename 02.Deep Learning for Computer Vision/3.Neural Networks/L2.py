#Your First Model
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

#creating model:
model = SimpleModel()
#Running the Model
x = torch.randn(3, 4)
y = model(x)
print(model)
print("\nInput shape :", x.shape)
print("Output shape:", y.shape)

#viewing Parameter
for param in model.parameters():
    print(param.shape)
