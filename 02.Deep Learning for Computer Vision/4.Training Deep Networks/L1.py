import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import random_split
from torch.utils.data import DataLoader
transform = transforms.ToTensor()
#download dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
print(type(train_dataset))
print(len(train_dataset))
print(f"Total images : {len(train_dataset)}")
image, label = train_dataset[3]

print(type(image))
print(image.shape)
print(label)

#see images:
import matplotlib.pyplot as plt

plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label : {label}")
plt.axis("off")
plt.show()

#dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
#splitting dataset:

train_size = 45000
val_size = 5000

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size]
)
print(type(train_dataset))
print(type(val_dataset))

print("Training samples :", len(train_dataset))
print("Validation samples:", len(val_dataset))
print(len(test_dataset))
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.title(labels[i].item())
    plt.axis("off")

plt.show()

#epoch
for images, labels in train_loader:
    print(images.shape)

batch_size = 4
for batch_idx, (images, labels) in enumerate(train_loader):

    print(f"Batch {batch_idx + 1}")

    if batch_idx == 4:
        break