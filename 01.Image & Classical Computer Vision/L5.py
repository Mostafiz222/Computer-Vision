#Load an image.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("01_Image_Basics/images/noodles.webp")
img = np.array(img)

print(img.shape)

#ask 2:Create:
#Original
import cv2

img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Half size
h, w = img.shape[:2]
half = cv2.resize(img, (w//2, h//2))
#Double size
h, w = img.shape[:2]
double = cv2.resize(img, (w*2, h*2))
#Display all three together.
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title(f"Original\n{img.shape}")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(half)
plt.title(f"Half\n{half.shape}")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(double)
plt.title(f"double\n{double.shape}")
plt.axis("off")
plt.tight_layout()
plt.show()
#Task 3:Compare:

#Nearest
nearest = cv2.resize(
    img,
    (500, 500),
    interpolation=cv2.INTER_NEAREST
)
#Bilinear
bilinear = cv2.resize(
    img,
    (500, 500),
    interpolation=cv2.INTER_LINEAR
)
#Bicubic
bicubic = cv2.resize(
    img,
    (500, 500),
    interpolation=cv2.INTER_CUBIC
)
#Lanczos
lanczos = cv2.resize(
    img,
    (500, 500),
    interpolation=cv2.INTER_LANCZOS4
)

#Display them in one figure.
images = [
    ("Nearest", nearest),
    ("Linear", bilinear),
    ("Cubic", bicubic),
    ("Lanczos", lanczos)
]

plt.figure(figsize=(12,8))

for i, (title, image) in enumerate(images):
    plt.subplot(2,2,i+1)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
# Task 4:Rotate:

# 90°
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(rot90)
plt.title("90 Degrees")
plt.axis("off")
plt.show()
# 180°
rot180 = cv2.rotate(img, cv2.ROTATE_180)

plt.imshow(rot180)
plt.title("180 Degrees")
plt.axis("off")
plt.show()
# 270°
rot270 = cv2.rotate(
    img,
    cv2.ROTATE_90_COUNTERCLOCKWISE
)

plt.imshow(rot270)
plt.title("Counter Clockwise")
plt.axis("off")
plt.show()
# 45°
h, w = img.shape[:2]

center = (w//2, h//2)

matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    img,
    matrix,
    (w, h)
)

plt.imshow(rotated)
plt.title("45 Degrees")
plt.axis("off")
plt.show()
# Task 5:Save one transformed image.
import cv2

# Save the raw rotated image matrix directly
cv2.imwrite(
    "01_Image_Basics/images/rotated.jpg",
    cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
)
