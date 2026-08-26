import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

image = Image.open("06.Advanced Computer Vision\coin.jpg")

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(image)

# Example bounding box
rect = patches.Rectangle(
    (80, 60),      # (x, y)
    180,           # width
    140,           # height
    linewidth=2,
    edgecolor='red',
    facecolor='none'
)

ax.add_patch(rect)
plt.axis("off")
plt.show()

#segmentation:
import numpy as np
import matplotlib.pyplot as plt

# Create an empty mask
mask = np.zeros((100, 100))

# Add a square object
mask[30:70, 30:70] = 1

plt.imshow(mask, cmap="gray")
plt.title("Binary Segmentation Mask")
plt.axis("off")
plt.show()