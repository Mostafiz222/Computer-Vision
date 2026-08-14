# Task 1
# Load the same image twice.
# Display both.
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("01_Image_Basics/images/noodles.webp")
img2 = cv2.imread("01_Image_Basics/images/noodles.webp")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img1, cmap="gray")
plt.title("Image1")
plt.axis("off")
plt.show()
plt.imshow(img2, cmap="gray")
plt.title("Image2")
plt.axis("off")
plt.show()
# Task 2
# Detect ORB features.
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
# Print
print(len(kp1))
print(len(kp2))

# Task 3
# Match using BFMatcher.
bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)
matches = bf.match(
    des1,
    des2
)
matches = sorted(
    matches,
    key=lambda x: x.distance
)
# Sort by distance.
# Print:
print(matches)
# Task 4
# Draw the best 30 matches.
matched = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:40],
    None,
    flags=2
)
# Display.
matched = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:30],
    None,
    flags=2
)

plt.figure(figsize=(15,8))
plt.imshow(matched)
plt.axis("off")
plt.show()
# Task 5
# Rotate one image by 30
h, w = img2.shape[:2]
matrix = cv2.getRotationMatrix2D(
    (w//2, h//2),
    30,
    1
)
rotated = cv2.warpAffine(
    img2,
    matrix,
    (w,h)
)
# Match again.
kp_rot, des_rot = orb.detectAndCompute(
    rotated,
    None
)
matches = bf.match(
    des1,
    des_rot
)
matches = sorted(
    matches,
    key=lambda x: x.distance
)
# Display.
matched = cv2.drawMatches(
    img1,
    kp1,
    rotated,
    kp_rot,
    matches[:40],
    None,
    flags=2
)
plt.figure(figsize=(15,8))
plt.imshow(matched)
plt.axis("off")
plt.show()
# Task 6
# Use KNN matching.
bf = cv2.BFMatcher(
    cv2.NORM_HAMMING
)
matches = bf.knnMatch(
    des1,
    des2,
    k=2
)
# Apply Lowe's ratio test.
good = []

for m, n in matches:

    if m.distance < 0.75 * n.distance:

        good.append(m)
# Display only good matches.

matched = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=2
)
plt.figure(figsize=(15,8))
plt.imshow(matched)
plt.axis("off")
plt.show()

# Task 7
# Compare:
# BFMatcher
# Ratio Test
# Which produces cleaner matches?
# The Ratio Test (applied on top of a Brute-Force Matcher using 
# k-nearest neighbors) produces significantly cleaner matches 
# than a standard BFMatcher alone.
# Why?
# Distinctiveness Check
# Ambiguity Rejection