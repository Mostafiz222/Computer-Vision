# Task 1
# Load an image.
# Convert it to grayscale.
# Display both.
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.show()
plt.imshow(gray, cmap="gray")
plt.title("Gray")
plt.axis("off")
plt.show()

# Task 2
# Apply:
# Sobel X
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)
# Sobel Y
sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)
# Display together
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")

plt.subplot(1,2,2)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")

plt.show()
# Task 3
# Compute the gradient magnitude.
# Display it.
magnitude = cv2.magnitude(
    sobel_x,
    sobel_y
)

plt.imshow(magnitude, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")
plt.show()
# Task 4
# Apply Laplacian.
# Display it.
lap = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

plt.imshow(lap, cmap="gray")
plt.title("Laplacian")
plt.axis("off")
plt.show()

# Task 5
# Apply Canny using:(50,150)
edges = cv2.Canny(
    gray,
    50,
    150
)

plt.imshow(edges, cmap="gray")
plt.title("Canny")
plt.axis("off")
plt.show()
# Apply Canny using:(100,200)
edges = cv2.Canny(
    gray,
    100,
    200
)

plt.imshow(edges, cmap="gray")
plt.title("Canny")
plt.axis("off")
plt.show()
# Compare the outputs.
#->(50,150) is more detailed compare to (100,200)

# Task 6
# Display all edge detectors in one figure.
images = [
    ("Gray", gray),
    ("Sobel X", sobel_x),
    ("Sobel Y", sobel_y),
    ("Magnitude", magnitude),
    ("Laplacian", lap),
    ("Canny", edges)
]

plt.figure(figsize=(15,8))

for i, (title, image) in enumerate(images):
    plt.subplot(2,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# Task 7
# Save the Canny output
cv2.imwrite(
    "01_Image_Basics/images/canny.png",
    edges
)