import torch
import torch.nn as nn

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
x = torch.randn(
    1,
    16,
    28,
    28
)
output = pool(x)
print(output.shape)

#First CNN:
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

model = SimpleCNN()
print("print model :",model)

images = torch.randn(
    64,
    1,
    28,
    28
)
output = model(images)
print(output.shape)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(total_params)