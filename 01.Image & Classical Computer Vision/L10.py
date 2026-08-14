# Task 1
# Load the Otsu image from the previous lesson.
# Display it.
import matplotlib.pyplot as plt
import cv2
import numpy as np
img = cv2.imread("01_Image_Basics/images/otsu.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.title("Otsu")
plt.axis("off")
plt.show()
# Task 2
# Create a 3×3 kernel.
# Print it.
kernel = np.ones((3,3), dtype=np.uint8)
print(kernel)
# Task 3
# Apply:
# Erosion
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(
    img,
    kernel,
    iterations=1
)
# Dilation
dilation = cv2.dilate(
    img,
    kernel,
    iterations=1
)
# Display all together.
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(erosion, cmap="gray")
plt.title("Erosion")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(dilation, cmap="gray")
plt.title("Dilation")
plt.axis("off")
plt.show()

# Task 4
# Apply:
# Opening
opening = cv2.morphologyEx(
    img,
    cv2.MORPH_OPEN,
    kernel
)
# Closing
closing = cv2.morphologyEx(
    img,
    cv2.MORPH_CLOSE,
    kernel
)
# Display together.
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(opening, cmap="gray")
plt.title("Opening")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(closing, cmap="gray")
plt.title("Closing")
plt.axis("off")
plt.show()

# Task 5
# Apply:
# Gradient
gradient = cv2.morphologyEx(
    img,
    cv2.MORPH_GRADIENT,
    kernel
)
# Top Hat
tophat = cv2.morphologyEx(
    img,
    cv2.MORPH_TOPHAT,
    kernel
)
# Black Hat
blackhat = cv2.morphologyEx(
    img,
    cv2.MORPH_BLACKHAT,
    kernel
)
# Display together.
images = [
    ("gradient",gradient),
    ("tophat", tophat),
    ("blackhat",blackhat)
]

plt.figure(figsize=(8,4))
for i, (title, image) in enumerate(images):
    plt.subplot(1,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# Task 6
# Try kernel sizes:
# 3×3
kernel = np.ones((3,3), np.uint8)
erosion_3 = cv2.erode(
    img,
    kernel,
    iterations=1
)
# Dilation
dilation_3 = cv2.dilate(
    img,
    kernel,
    iterations=1
)
# 7×7
kernel = np.ones((7,7), np.uint8)
erosion_7 = cv2.erode(
    img,
    kernel,
    iterations=1
)
# Dilation
dilation_7 = cv2.dilate(
    img,
    kernel,
    iterations=1
)
# Compare erosion and dilation.
images = [
    ("erosion_3",erosion_3),
    ("dilation_3", dilation_3),
    ("erosion_7",erosion_7),
    ("dilation_7", dilation_7),

]

plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(2,2,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# What changes?
#using kernel 7*7 made the img more darker compare to 3*3 in erosion.same case in dialation. 
#the 7*7 in more white than 3*3 kernel

# Task 7
# Save your favorite result.
cv2.imwrite(
    "01_Image_Basics/images/opening.png",
    opening
)