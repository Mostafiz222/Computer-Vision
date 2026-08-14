# Task 1
# Load an image.
# Convert it to grayscale.
# Display both.

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
images = [
    ("img",img),
    ("Gray", gray)
]
plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(3,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# Task 2
# Apply:
# Binary
_, binary = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_BINARY
)
# Binary Inverse
_, binary_inv = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_BINARY_INV
)
# Display together.
images = [
    ("binary",binary),
    ("binary_inv", binary_inv)
]

plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(3,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Task 3
# Apply:
# Trunc
_, trunc = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_TRUNC
)
# To Zero
_, tozero = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_TOZERO
)
# Display together.
images = [
    ("trunc",trunc),
    ("tozero", tozero)
]

plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(3,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()


# Task 4
# Apply:
# Adaptive Gaussian Threshold
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
# Otsu Threshold
_, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
# Display together.
images = [
    ("adaptive",adaptive),
    ("otsu", otsu)
]

plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(3,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Task 5
# Display all thresholding methods in one figure.

images = [
    ("Gray", gray),
    ("Binary", binary),
    ("Binary Inv", binary_inv),
    ("Trunc", trunc),
    ("To Zero", tozero),
    ("Adaptive", adaptive),
    ("Otsu", otsu)
]

plt.figure(figsize=(15,10))

for i, (title, image) in enumerate(images):
    plt.subplot(3,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# Task 6
# Print the threshold value selected by Otsu.
threshold, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu Threshold:", threshold)
# Task 7
# Save the Otsu result.
cv2.imwrite(
    "01_Image_Basics/images/otsu.png",
    otsu
)