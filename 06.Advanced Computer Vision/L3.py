from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

domains = {
    "Original": transforms.Compose([]),
    "Bright": transforms.ColorJitter(brightness=0.8),
    "Blur": transforms.GaussianBlur(kernel_size=5),
    "Gray": transforms.Grayscale(num_output_channels=3),
}

plt.figure(figsize=(12,3))

for i, (name, transform) in enumerate(domains.items()):
    img = transform(image)
    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()


#OOD
import torch
import torch.nn.functional as F

# Simulated embeddings
train_embeddings = F.normalize(torch.randn(100, 512), dim=1)

id_sample = F.normalize(train_embeddings[0] + 0.02 * torch.randn(512), dim=0)
ood_sample = F.normalize(torch.randn(512), dim=0)

# Cosine similarity
id_similarity = torch.matmul(train_embeddings, id_sample).max()
ood_similarity = torch.matmul(train_embeddings, ood_sample).max()

print(f"Nearest similarity (ID):  {id_similarity:.4f}")
print(f"Nearest similarity (OOD): {ood_similarity:.4f}")


#Robustness

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

corruptions = {
    "Original": transforms.Compose([]),
    "Gaussian Blur": transforms.GaussianBlur(7),
    "Brightness": transforms.ColorJitter(brightness=0.4),
    "Grayscale": transforms.Grayscale(num_output_channels=3),
}

plt.figure(figsize=(12,3))

for i, (name, transform) in enumerate(corruptions.items()):
    img = transform(image)

    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()

#XAI
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load pretrained model
model = models.resnet18(weights="DEFAULT")
model.eval()

# Load image
image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = transform(image).unsqueeze(0)
x.requires_grad = True

# Forward pass
output = model(x)

pred = output.argmax()

# Backward pass
output[0, pred].backward()

# Saliency map
saliency = x.grad.abs().max(dim=1)[0]

plt.imshow(saliency.squeeze().detach().numpy(), cmap="hot")
plt.axis("off")
plt.title("Saliency Map")
plt.show()