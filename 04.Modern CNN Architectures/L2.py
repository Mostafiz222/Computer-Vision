import torch
import torch.nn as nn
#Inception-style Branch
# 1. Define channel count and dummy input tensor (Batch, Channels, Height, Width)
in_channels = 32
x = torch.randn(1, in_channels, 28, 28)

# 2. Instantiate Conv2d layers
b1 = nn.Conv2d(in_channels, 64, kernel_size=1)
b2 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
b3 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)

# 3. Pass tensor x through each layer
out1 = b1(x)
out2 = b2(x)
out3 = b3(x)

# 4. Concatenate output tensors along channel dimension (dim=1)
output = torch.cat([out1, out2, out3], dim=1)

print(output.shape)  # Output shape: torch.Size([1, 192, 28, 28])

#Residual Connection

import torch
import torch.nn as nn

# Define a standard Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

# Initialize dummy input: (Batch_Size=2, Channels=64, Height=32, Width=32)
x = torch.randn(2, 64, 32, 32)

# Case 1: Same channels and stride
block_same = ResidualBlock(in_channels=64, out_channels=64, stride=1)
output_same = block_same(x)

# Case 2: Downsampling channels and spatial size (stride=2, channels 64 -> 128)
block_down = ResidualBlock(in_channels=64, out_channels=128, stride=2)
output_down = block_down(x)

print(f"Input Shape:            {x.shape}")
print(f"Same Dimension Output:  {output_same.shape}")
print(f"Downsampled Output:     {output_down.shape}")
#Dense Connection
import torch
import torch.nn as nn

class SimpleDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Each layer takes (in_channels + i * growth_rate) and outputs growth_rate channels
        for i in range(num_layers):
            layer_in = in_channels + (i * growth_rate)
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_in),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(layer_in, growth_rate, kernel_size=3, padding=1, bias=False)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all prior feature maps along channel axis
            concatenated_input = torch.cat(features, dim=1)
            new_feature = layer(concatenated_input)
            features.append(new_feature)
            
        return torch.cat(features, dim=1)

# Dummy Input: Batch=2, Channels=16, Height=32, Width=32
x = torch.randn(2, 16, 32, 32)

# Block with 3 layers, each adding 12 new channels (growth_rate = 12)
dense_block = SimpleDenseBlock(in_channels=16, growth_rate=12, num_layers=3)
output = dense_block(x)

print(f"Input Shape:  {x.shape}")
print(f"Output Shape: {output.shape}")