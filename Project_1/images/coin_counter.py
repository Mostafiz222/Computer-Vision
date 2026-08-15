# Task 1
# Complete the full pipeline.
#Step 1 — Load Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("project_1/images/coin.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.show()
#Step 2 — Convert to Gray
gray = cv2.cvtColor(
    img,
    cv2.COLOR_RGB2GRAY
)
#Step 3 — Blur
blur = cv2.GaussianBlur(
    gray,
    (5,5),
    0
)
#Step 4 — Otsu Threshold
_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
#Step 5 — Morphology
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)
#Step 6 — Find Contours
contours, _ = cv2.findContours(
    opening,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
#Step 7 — Count Objects
print("Objects Found:", len(contours))
#Step 8 — Draw Circles
result = img.copy()

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 2000:
        continue

    (x, y), radius = cv2.minEnclosingCircle(cnt)

    center = (int(x), int(y))

    radius = int(radius)

    cv2.circle(
        result,
        center,
        radius,
        (255,0,0),
        2
    )
#Step 9 — Number the Coins
count = 1
for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 2000:
        continue

    (x,y), radius = cv2.minEnclosingCircle(cnt)

    center = (int(x), int(y))

    cv2.putText(
        result,
        str(count),
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    count += 1
#Step 10 — Statistics
areas = []

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 2000:

        areas.append(area)

print("Coin Count:", len(areas))

print("Largest Area:", max(areas))

print("Smallest Area:", min(areas))

print("Average Area:", sum(areas)/len(areas))
#Step 11 — Display Final Result
plt.figure(figsize=(8,8))

plt.imshow(result)

plt.axis("off")

plt.title("Detected Coins")

plt.show()
# Task 2
# Print area of each coin
# Combined processing loop with individual area outputs
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2000:
        continue

    areas.append(area)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(result, center, radius, (255, 0, 0), 2)
    print(f"Coin {count} Area: {area:.2f}")
    cv2.putText(
        result,
        str(count),
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    count += 1

# Task 3
# Draw
# Bounding rectangle
# Enclosing circle
# on the same object.
# Combined processing loop drawing both shapes
result = img.copy()
areas = []
count = 1
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2000:
        continue

    areas.append(area)
    (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x_c), int(y_c))
    radius = int(radius)
    cv2.circle(
        result, center, radius, (255, 0, 0), 2
    ) 
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(
        result, (x, y), (x + w, y + h), (0, 0, 255), 2
    )
    print(f"Coin {count} Area: {area:.2f}")
    cv2.putText(
        result,
        str(count),
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    count += 1

# Display the final figure
plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.title("Task 3: Bounding Rectangles & Enclosing Circles")
plt.axis("off")
plt.show()
# Task 4
# Use different colors for each detected object.
# Count valid contours for uniform hue spacing
result = img.copy()
areas = []
count = 1
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 2000]
num_coins = len(valid_contours)

for cnt in valid_contours:
    area = cv2.contourArea(cnt)
    areas.append(area)

    hue = int((count - 1) * (180 / max(num_coins, 1))) % 180
    hsv_pixel = np.uint8([[[hue, 255, 255]]])
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
    color = (int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2]))

    (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x_c), int(y_c))
    cv2.circle(result, center, int(radius), color, 2)

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        result,
        str(count),
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    print(f"Coin {count} Area: {area:.2f}")
    count += 1

plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.title("Task 4: Distinct Colors per Object")
plt.axis("off")
plt.show()
# Task 5
# Print
# Total Coins
# Average Area
# Largest Coin
# Smallest Coin
# Print individual coin details and compute overall summary statistics
if areas:
    print("-" * 30)
    print(f"Total Coins: {len(areas)}")
    print(f"Average Area: {sum(areas) / len(areas):.2f}")
    print(f"Largest Coin: {max(areas):.2f}")
    print(f"Smallest Coin: {min(areas):.2f}")
    print("-" * 30)
else:
    print("No valid coins detected based on the area threshold.")
