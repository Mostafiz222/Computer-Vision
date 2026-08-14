# Task 1 — Create Tensors
import torch
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3, 4])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor3d = torch.rand(3, 4, 5)
print("--- Scalar ---")
print("Value:", scalar)
print("Shape:", scalar.shape)
print("\n--- Vector ---")
print("Value:", vector)
print("Shape:", vector.shape)
print("\n--- 2x3 Matrix ---")
print("Value:\n", matrix)
print("Shape:", matrix.shape)
print("\n--- 3D Tensor ---")
print("Value:\n", tensor3d)
print("Shape:", tensor3d.shape)
#Task 2 — Create from NumPy
import numpy as np
arr = np.array([10, 20, 30, 40, 50])
# Convert NumPy array to a PyTorch tensor
tensor = torch.from_numpy(arr)
print("--- NumPy Array ---")
print("Value:", arr)
print("Type: ", type(arr))
print("\n--- PyTorch Tensor ---")
print("Value:", tensor)
print("Type: ", type(tensor))

# Task 3 — Convert Back
# Convert the tensor back to NumPy.
# Print the result.
# Convert PyTorch tensor back to a NumPy array
arr_converted = tensor.numpy()
print("--- Converted back to NumPy ---")
print("Value:", arr_converted)
print("Type: ", type(arr_converted))

#Task 4 — Random Image Tensor
import torch

# Create random image tensor (3 channels, 224 height, 224 width)
image_tensor = torch.rand(3, 224, 224)
# Print requested properties
print("Shape:    ", image_tensor.shape)
print("Dtype:    ", image_tensor.dtype)
print("Min Value:", image_tensor.min().item())
print("Max Value:", image_tensor.max().item())

#Task 5 — Batch of Images
import torch
# Create a batch of 16 RGB images of size 224x224
image_batch = torch.rand(16, 3, 224, 224)
# Print its shape
print("Batch Shape:", image_batch.shape)

#Task 6 — Reshape

# Create initial 1D tensor from 0 to 11
x = torch.arange(12)

# Reshape into 3x4
x_3x4 = x.reshape(3, 4)

# Reshape into 2x2x3
x_2x2x3 = x.reshape(2, 2, 3)

# Print everything
print("--- Original (1D) ---")
print("Value:", x)
print("Shape:", x.shape)

print("\n--- Reshaped (3x4) ---")
print("Value:\n", x_3x4)
print("Shape:", x_3x4.shape)

print("\n--- Reshaped (2x2x3) ---")
print("Value:\n", x_2x2x3)
print("Shape:", x_2x2x3.shape)

#Task7:Permute


# 1. Create fake image in (Height, Width, Channels) format: 224 x 224 x 3
img_hwc = torch.rand(224, 224, 3)

# 2. Permute dimensions: index 2 (Channels) becomes 0, index 0 (H) becomes 1, index 1 (W) becomes 2
img_chw = img_hwc.permute(2, 0, 1)

# Print both shapes
print("Original Shape (HWC):", img_hwc.shape)
print("Permuted Shape (CHW):", img_chw.shape)