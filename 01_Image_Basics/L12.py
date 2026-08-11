# Task 1
# Load your image.
# Convert to grayscale.
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("01_Image_Basics/images/noodles.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original = img
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img, cmap="gray")
plt.title("gray")
plt.axis("off")

plt.show()
# Task 2
# Apply Harris.
# Display.
img = cv2.imread(
    "01_Image_Basics/images/noodles.webp"
)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)
corners = cv2.cornerHarris(
    gray,
    2,
    3,
    0.04
)
img[corners > 0.01 * corners.max()] = [255,0,0]
Harris=img
plt.imshow(img)
plt.axis("off")
plt.show()
# Task 3
# Apply Shi-Tomasi.
# Draw circles.
corners = cv2.goodFeaturesToTrack(
    gray.astype("uint8"),
    100,
    0.01,
    10
)
for corner in corners:

    x, y = corner.ravel()

    cv2.circle(
        img,
        (int(x), int(y)),
        4,
        (255,0,0),
        -1
    )
Shi_Tomasi=img
plt.imshow(img)
plt.axis("off")
plt.show()

# Task 4
# Apply FAST.
# Draw keypoints.
# Task 4
# Apply FAST.
# Draw keypoints.

gray = gray.astype("uint8")
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(gray, None)
img = cv2.cvtColor(cv2.imread("01_Image_Basics/images/noodles.webp"), cv2.COLOR_BGR2RGB)
kp_fast=kp
img_fast = cv2.drawKeypoints(
    img,
    kp,
    None,
    color=(255, 0, 0)
)
FAST=img_fast
plt.imshow(img_fast)
plt.axis("off")
plt.show()

# Task 5
# Apply ORB.
# Draw keypoints.
orb = cv2.ORB_create()

kp = orb.detect(gray, None)
kp_orb =kp
kp, des = orb.compute(gray, kp)

img_orb = cv2.drawKeypoints(
    img,
    kp,
    None,
    color=(255,0,0)
)
ORB =img_orb
plt.imshow(img_orb)
plt.axis("off")
plt.show()
# Task 6

# Display
# Original
# Harris
# Shi-Tomasi
# FAST
# ORB
# in one figure.
images = [
    ("original",original),
    ("Harris", Harris),
    ("Shi_Tomasi",Shi_Tomasi),
    ("FAST", FAST),
    ("ORB",ORB)

]
plt.figure(figsize=(10,10))
for i, (title, image) in enumerate(images):
    plt.subplot(2,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Task 7
# Print
print("FAST:", len(kp_fast))
print("ORB :", len(kp_orb))