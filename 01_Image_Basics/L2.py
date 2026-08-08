#import numpy as np
#import matplotlib.pyplot as plt

# img = np.zeros((100, 100), dtype=np.uint8)

# # Create a white square
# img[25:75, 25:75] = 255

# plt.imshow(img, cmap="gray")
# plt.show()

# #Gradient

# img = np.zeros((256, 256), dtype=np.uint8)

# for i in range(256):
#     img[i, :] = i

# plt.imshow(img, cmap="gray")
# plt.show()

# #Color image:
# img = np.zeros((100, 100, 3), dtype=np.uint8)

# # Green square
# img[25:75, 25:75] = [0, 255, 0]

# plt.imshow(img)
# plt.show()
#from PIL import Image

# img = Image.open("01_Image_Basics/images/noodles.webp")

# print("Size:", img.size)
# print("Mode:", img.mode)

# plt.imshow(img)
# plt.axis("off")
# plt.show()
#Convert Pillow → NumPy

# img = Image.open("01_Image_Basics/images/noodles.webp")

# img_array = np.array(img)

# print(type(img))
# print(type(img_array))

# print(img_array.shape)
# print(img_array.dtype)
# pixel = img_array[100, 200]

# print(pixel)
# print(img_array[100, 200, 0])  # Red
# print(img_array[100, 200, 1])  # Green
# print(img_array[100, 200, 2])  # Blue

# Practice Q1:Create a black 100 × 100 image.
import numpy as np
import matplotlib.pyplot as plt
img = np.zeros((100, 100), dtype=np.uint8)
plt.imshow(img, cmap="gray")
plt.show()

# Practice Q2:Create a white square in its center.
import numpy as np
import matplotlib.pyplot as plt
img = np.zeros((100, 100), dtype=np.uint8)
img[25:75, 25:75] = 255
plt.imshow(img, cmap="gray")
plt.show()
# Practice Q3:Create a grayscale gradient.
img = np.zeros((256, 256), dtype=np.uint8)

for i in range(256):
    img[i, :] = i

plt.imshow(img, cmap="gray")
plt.show()
#Practice Q4:Create a 100 × 100 × 3 RGB image containing four colored squares:
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[0:50, 0:50] = [255, 0, 0]
img[0:50, 50:100] = [0, 255, 0]
img[50:100, 0:50] = [0, 0, 255]
img[50:100, 50:100] = [255, 255, 0]
plt.imshow(img)
plt.show()
#Practice Q5:Load your real image using Pillow.
from PIL import Image

img = Image.open("01_Image_Basics/images/noodles.webp")

print("Size:", img.size)
print("Mode:", img.mode)
img_array = np.array(img)
print("Size:", img_array.size)
print("dtype:", img_array.dtype)
plt.imshow(img)
plt.axis("off")
plt.show()

#Practice Question 6:Print the RGB value of three different pixels.

pixel = img_array[100, 200]
print(pixel)
pixel = img_array[150, 250]
print(pixel)
pixel = img_array[190, 210]
print(pixel)