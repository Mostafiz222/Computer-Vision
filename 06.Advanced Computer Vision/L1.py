import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Load your own image
image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

# Define augmentation pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.GaussianBlur(kernel_size=5)
])

# Generate augmented views
views = [transform(image) for _ in range(4)]

# Display results
plt.figure(figsize=(12, 3))
for i, view in enumerate(views):
    plt.subplot(1, 4, i + 1)
    plt.imshow(view)
    plt.axis("off")
plt.tight_layout()
plt.show()


import torch
import torch.nn.functional as F

# Example embeddings
anchor = torch.randn(512)
positive = anchor + 0.05 * torch.randn(512)   # Similar vector
negative = torch.randn(512)                   # Unrelated vector

# Normalize embeddings
anchor = F.normalize(anchor, dim=0)
positive = F.normalize(positive, dim=0)
negative = F.normalize(negative, dim=0)

# Compute cosine similarity
pos_sim = F.cosine_similarity(anchor, positive, dim=0)
neg_sim = F.cosine_similarity(anchor, negative, dim=0)

print(f"Positive similarity: {pos_sim:.4f}")
print(f"Negative similarity: {neg_sim:.4f}")