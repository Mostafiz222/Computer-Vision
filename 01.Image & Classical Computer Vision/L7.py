# Task 1
# Load an image.
# Display it.
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.show()

# Task 2
# Apply
# Mean Filter
mean = cv2.blur(img, (5,5))
# Gaussian Blur
gaussian = cv2.GaussianBlur(
    img,
    (5,5),
    0
)
# Median Filter
median = cv2.medianBlur(
    img,
    5
)
# Bilateral Filter
bilateral = cv2.bilateralFilter(
    img,
    9,
    75,
    75
)

# Task 3:Display all five images together.
images = [
    ("Original", img),
    ("Mean", mean),
    ("Gaussian", gaussian),
    ("Median", median),
    ("Bilateral", bilateral)
]

plt.figure(figsize=(15,8))

for i,(title,image) in enumerate(images):
    plt.subplot(2,3,i+1)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Task 4:Increase kernel size.
l_mean = cv2.blur(img, (7,7))
plt.imshow(l_mean)
plt.title("Large Kernel")
plt.axis("off")
plt.show()
# Task 5
# Save your favorite filtered image.
cv2.imwrite(
    "01_Image_Basics/images/gaussian.jpg",
    cv2.cvtColor(gaussian, cv2.COLOR_RGB2BGR)
)