#Task 1:Load your image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("01_Image_Basics/images/noodles.webp")
img = np.array(img)

plt.imshow(img)
plt.axis("off")
plt.show()

#Task 2:Print:
print(img.shape)
#Task 3:Crop:Top-left/Center/Bottom-right/Display each crop.
#top-left
crop = img[0:300, 0:300]
plt.imshow(crop)
plt.axis("off")
plt.show()
#center
crop = img[200:600, 220:520]
plt.imshow(crop)
plt.axis("off")
plt.show()
#Bottom_right
crop = img[400:810, 300:648]
plt.imshow(crop)
plt.axis("off")
plt.show()

#Task 4:Copy one crop and paste it into another location.
roi = img[100:200, 100:200].copy()
img[300:400, 300:400] = roi

plt.imshow(img)
plt.axis("off")
plt.show()

#Task 5:Create:Horizontal flip/Vertical flip
horizontal = img[:, ::-1]

plt.imshow(horizontal)
plt.title("Horizontal Flip")
plt.axis("off")
plt.show()

vertical = img[::-1, :]

plt.imshow(vertical)
plt.title("Vertical Flip")
plt.axis("off")
plt.show()

#Task 6:Using OpenCV:Draw:
#One rectangle
import cv2
opencv_img = cv2.imread("01_Image_Basics/images/noodles.webp")
opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

cv2.rectangle(
    opencv_img,
    (50, 50),       # top-left
    (250, 250),     # bottom-right
    (255, 0, 0),    # red
    3               # thickness
)
plt.imshow(opencv_img)
plt.axis("off")
plt.show()
#One circle
cv2.circle(
    opencv_img,
    (200, 200),
    60,
    (0, 255, 0),
    3
)

plt.imshow(opencv_img)
plt.axis("off")
plt.show()
#One line
cv2.line(
    opencv_img,
    (0, 0),
    (400, 300),
    (255, 255, 0),
    3
)

plt.imshow(opencv_img)
plt.axis("off")
plt.show()
#ask 7:Write your name somewhere on the image.
cv2.putText(
    opencv_img,
    "And I am Iron Man",
    (50, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

plt.imshow(opencv_img)
plt.axis("off")
plt.show()