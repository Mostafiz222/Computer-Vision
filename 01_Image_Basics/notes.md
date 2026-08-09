Lesson1:
This lesson is all about how an Image is represented in a 2D matrix.
Grayscale image: Black and white(0-1).Uses 8 bit unsigned integer (0-255).Represents  different shades of grey.
Color image: 3 color channels is used(R,G,B).
Image shape: imagine height=5,width= 5.
Grayscale image shape=[5,5]
color image shape=[5,5,3]
Lesson2:
An image isn't an abstract object to a computer. It is structured numerical data, and by changing those numbers we can directly change the image.
Lesson 3:RGB vs BGR & Understanding Image Channels
OpenCv uses by default BGR ,we can convert it to RGB
Lesson 4:
Research Connection:
Bounding boxes in object detection (YOLO, Faster R-CNN, DETR) are fundamentally just rectangles drawn on regions of interest.what's happening under the hood is that the model predicts the coordinates of a rectangle, and then code very similar to cv2.rectangle() draws it on the image.
lesson 5:
Interpolation"This is one of the most important concepts today.
When resizing, new pixels must be created or removed.
How should the computer estimate them?
That's called interpolation.
OpenCV provides several methods
