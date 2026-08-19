#Convoulation:without padding
import torch
import torch.nn as nn

image = torch.randn(1,1,28,28)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3
)

output = conv(image)
print(image.shape)
print(output.shape)

##with padding:
conv = nn.Conv2d(
    1,
    16,
    kernel_size=3,
    padding=1
)
print("With Padding:")
output = conv(image)
print(output.shape)

#stride:
conv = nn.Conv2d(
    1,
    16,
    kernel_size=3,
    stride=2,
    padding=1
)
print("stride:")
output = conv(image)
print(output.shape)