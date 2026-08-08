import numpy as np
#crating image matrix:
img = np.array([
    [0, 128, 240],
    [120, 5, 230],
    [10, 20, 190]
],dtype= np.uint8)
#print image matrix:
print(img)
print("shape :", img.shape )
print("Data type :" ,img.dtype)
print("Image Size :", img.size)