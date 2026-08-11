# Task 1
# Load the Otsu image.
# Display it.
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(
    "01_Image_Basics/images/otsu.png",
    cv2.IMREAD_GRAYSCALE
)

plt.imshow(img, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.show()


# Task 2
# Find contours.
contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
# Task 3
# Draw all contours.
# Display them.
color = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2RGB
)
cv2.drawContours(
    color,
    contours,
    -1,
    (255,0,0),
    2
)
plt.imshow(color)
plt.title("Contours")
plt.axis("off")
plt.show()

# Task 4
# For every contour print:
print("Number of Objects:", len(contours))
# Area
print("Number of Objects:", len(contours))
for i, contour in enumerate(contours):

    area = cv2.contourArea(contour)

    print(f"Contour {i}: Area = {area}")
# Perimeter
for contour in contours:

    perimeter = cv2.arcLength(
        contour,
        True
    )

    print(perimeter)



# Task 5
# Draw bounding rectangles.
# Display.
color = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2RGB
)

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(
        color,
        (x,y),
        (x+w,y+h),
        (0,255,0),
        2
    )

plt.imshow(color)
plt.axis("off")
plt.show()

# Task 6
# Draw enclosing circles.
# Display
color = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2RGB
)

for contour in contours:

    (x,y), radius = cv2.minEnclosingCircle(contour)

    center = (int(x), int(y))

    radius = int(radius)

    cv2.circle(
        color,
        center,
        radius,
        (255,0,0),
        2
    )

plt.imshow(color)
plt.axis("off")
plt.show()


# Task 7
# Draw convex hulls.
# Display
color = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2RGB
)

for contour in contours:

    hull = cv2.convexHull(contour)

    cv2.drawContours(
        color,
        [hull],
        -1,
        (255,0,0),
        2
    )

plt.imshow(color)
plt.axis("off")
plt.show()

import cv2
import matplotlib.pyplot as plt
#Task 8 (Small Challenge):Ignore tiny contours.
color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100:
        continue
    cv2.drawContours(color, [cnt], -1, (255, 0, 0), 2)

plt.imshow(color)
plt.title("Contours")
plt.axis("off")
plt.show()