#Task 1
# Load an image.
# Print its shape.
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

# Task 2
# Create
# Original
# Bright (+50)
# Dark (-50)
# Display all together.



bright = cv2.convertScaleAbs(
    img,
    alpha=1,
    beta=50
)

dark = cv2.convertScaleAbs(
    img,
    alpha=1,
    beta=-50
)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(bright)
plt.title("Bright")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(dark)
plt.title("Dark")
plt.axis("off")

plt.show()

# Task 3
# Create
# Low contrast
# High contrast
# Display all together.

high = cv2.convertScaleAbs(
    img,
    alpha=1.8,
    beta=0
)

low = cv2.convertScaleAbs(
    img,
    alpha=0.5,
    beta=0
)
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(high)
plt.title("High Contrast")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(low)
plt.title("Low Contrast")
plt.axis("off")
plt.show()
# Task 4
# Convert to grayscale.
# Plot its histogram.
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(8,5))
plt.hist(
    gray.ravel(),
    bins=256,
    range=(0,256)
)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()
# Task 5
# Apply
# Histogram Equalization
# CLAHE
# Display
equalized = cv2.equalizeHist(gray)
plt.figure(figsize=(10,4))
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)
clahe_img = clahe.apply(gray)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(gray,cmap="gray")
plt.title("Original")
plt.subplot(1,3,2)
plt.imshow(equalized,cmap="gray")
plt.title("Equalized")
plt.subplot(1,3,3)
plt.imshow(clahe_img,cmap="gray")
plt.title("CLAHE")
plt.show()
# Task 6
# Plot histograms for
# Original grayscale
# Equalized image
# CLAHE image
# Compare them.

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(
    gray.ravel(),
    bins=256,
    range=(0,256)
)
plt.title("Original")
plt.subplot(1,3,2)
plt.hist(
    equalized.ravel(),
    bins=256,
    range=(0,256)
)
plt.title("Equalized")
plt.subplot(1,3,3)
plt.hist(
    clahe_img.ravel(),
    bins=256,
    range=(0,256)
)
plt.title("CLAHE")
plt.show()