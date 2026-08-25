from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
])

view1 = transform(image)
view2 = transform(image)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(view1)
plt.title("View 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(view2)
plt.title("View 2")
plt.axis("off")

plt.show()


from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

image = Image.open("06.Advanced Computer Vision\coin.jpg").convert("RGB")

bright = transforms.ColorJitter(brightness=0.8)(image)
dark = transforms.ColorJitter(brightness=0.2)(image)
blur = transforms.GaussianBlur(kernel_size=7)(image)

images = [image, bright, dark, blur]
titles = ["Original", "Bright", "Dark", "Blurred"]

plt.figure(figsize=(12,3))

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()