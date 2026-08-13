#Task 1 — Tensor Indexing
import torch
x = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(x[0,0])
#Task 2 — Image Tensor
# Create a random image tensor with shape (3, 224, 224) -> (Channels, Height, Width)
img = torch.rand(3, 224, 224)
red = img[0]
green = img[1]
blue = img[2]
pixel_rgb = img[:, 100, 100]
print("Overall Shape:        ", img.shape)
print("Red Channel Shape:   ", red.shape)
print("Green Channel Shape: ", green.shape)
print("Blue Channel Shape:  ", blue.shape)
print("\nRGB values at pixel (100, 100):")
print(f"  Red:   {pixel_rgb[0].item():.4f}")
print(f"  Green: {pixel_rgb[1].item():.4f}")
print(f"  Blue:  {pixel_rgb[2].item():.4f}")
#Task 3 — Arithmetic
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("a + b = ", a + b)
print("a - b = ", a - b)
print("a * b = ", a * b)
print("a / 2 = ", a / 2)
#Task 4 — Broadcasting
matrix = torch.ones(3, 3)
vector = torch.tensor([1, 2, 3])
result = matrix + vector
print("Matrix:\n", matrix)
print("\nVector:\n", vector)
print("\nResult (matrix + vector):\n", result)
#This addition works because of PyTorch's broadcasting rules,
# which allow tensor operations on shapes that aren't identical without duplicating data in memory.
#Task 5 — Statistics
x = torch.rand(5, 5)
print("Tensor:\n", x)
print("\n--- Statistics ---")
print("Max: ", x.max().item())
print("Min: ", x.min().item())
print("Mean:", x.mean().item())
print("Std: ", x.std().item())
print("Sum: ", x.sum().item())
#Task 6 — Brightness Adjustment
img = torch.rand(3, 224, 224)
brighter = torch.clamp(img + 0.2, min=0.0, max=1.0)
darker = torch.clamp(img - 0.2, min=0.0, max=1.0)
print(f"Original min/max: {img.min().item():.4f} / {img.max().item():.4f}")
print(f"Bright   min/max: {brighter.min().item():.4f} / {brighter.max().item():.4f}")
print(f"Dark     min/max: {darker.min().item():.4f} / {darker.max().item():.4f}")
#Task 7 — Channel Mean
img = torch.rand(3, 224, 224)
# Method 1: Compute mean along spatial dimensions (dim=(1, 2))
channel_means = img.mean(dim=(1, 2))
print("Method 1 (Tensor of channel means):")
print("Channel Means:", channel_means)
# Method 2: Slicing individual channels directly
red_mean = img[0].mean().item()
green_mean = img[1].mean().item()
blue_mean = img[2].mean().item()
print("\nMethod 2 (Individual scalar values):")
print(f"Red Channel Mean:   {red_mean:.4f}")
print(f"Green Channel Mean: {green_mean:.4f}")
print(f"Blue Channel Mean:  {blue_mean:.4f}")

