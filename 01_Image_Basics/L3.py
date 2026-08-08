import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((300, 300, 3), dtype=np.uint8)

# # # Red square
img[:, :] = [255, 0, 0]
plt.imshow(img)
plt.title("Red Image")
plt.axis("off")
plt.show()
# # # Green square
img[:, :] = [0, 255, 0]
plt.imshow(img)
plt.title("Green Image")
plt.axis("off")
plt.show()
# # # Blue square
img[:, :] = [0, 0, 255]
plt.imshow(img)
plt.title("Blue Image")
plt.axis("off")
plt.show()
# # # White sqare
img[:, :] = [255, 255, 255]
plt.imshow(img)
plt.title("White Image")
plt.axis("off")
plt.show()

#Task 2:Load a real image using Pillow.
from PIL import Image
img = Image.open("01_Image_Basics/images/noodles.webp")
img = np.array(img)
print(img.shape)
plt.imshow(img)
plt.axis("off")
plt.show()

# #Task 3:Split the image
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(red, cmap="gray")
plt.title("Red Channel")

plt.subplot(1,3,2)
plt.imshow(green, cmap="gray")
plt.title("Green Channel")

plt.subplot(1,3,3)
plt.imshow(blue, cmap="gray")
plt.title("Blue Channel")

plt.show()

#Task 4:Create:Image without red/Image without green/Image without blue
img_no_red = img.copy()
img_no_red[:, :, 0] = 0
plt.imshow(img_no_red)
plt.axis("off")
plt.show()
img_no_green = img.copy()
img_no_green[:, :, 1] = 0
plt.imshow(img_no_green)
plt.axis("off")
plt.show()
img_no_blue = img.copy()
img_no_blue[:, :, 2] = 0
plt.imshow(img_no_blue)
plt.axis("off")
plt.show()
#Task 5:Create:Red-only image/Green-only image/Blue-only image
red_only = img.copy()

red_only[:, :, 1] = 0

red_only[:, :, 2] = 0

plt.imshow(red_only)
plt.axis("off")
plt.show()
green_only = img.copy()

green_only[:, :, 0] = 0

green_only[:, :, 2] = 0

plt.imshow(green_only)
plt.axis("off")
plt.show()
blue_only = img.copy()

blue_only[:, :, 0] = 0

blue_only[:, :, 1] = 0

plt.imshow(blue_only)
plt.axis("off")
plt.show()

#ask 6:Load the same image using OpenCV.
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("01_Image_Basics/images/noodles.webp")

plt.imshow(img)
plt.axis("off")
plt.show()

#Task 7:Convert BGR → RGB.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()