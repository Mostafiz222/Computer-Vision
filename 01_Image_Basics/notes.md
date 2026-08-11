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
lesson 7:
Convolution is sliding a small kernel over an image and combining the kernel values with the surrounding pixels to produce a new image. Depending on the kernel, it can blur, sharpen, detect edges, or extract features.(like we studied in academic)
lesson 8:
->Edge detectors calculate changes in pixel intensity. A grayscale image has one intensity value per pixel, making the computation simpler and faster than processing three separate RGB channels.
->Laplacian directly computes the second derivative of the image. Second derivatives respond strongly to small intensity fluctuations, including random noise. Canny, on the other hand, first applies Gaussian smoothing to reduce noise before detecting edges, making it much more robust.
lesson 9:Thresholding converts an image into binary regions based on absolute pixel intensity values, whereas edge detection identifies boundaries by locating sharp changes or gradients in local pixel intensity.
Lesson 10:
What are Morphological Operations?
Morphology means:
Operations that modify the shape of objects in a binary image.
Unlike brightness or filtering, morphology focuses on the geometry of foreground objects.
Question 1:
Suppose you have tiny white noise scattered across a binary image.
Which operation would you use?
Opening
Closing
Why?
->Opening
->Erosion step: Erases small isolated white pixels (noise) because the structuring element does not fit entirely within these tiny spots.
->Dilation step: Restores the size and shape of the larger foreground objects that survived the erosion.
Question 2:Suppose a white object contains many tiny black holes.
Which operation would you choose?
Why?
->Closing.
->Dilation step: Expands the white foreground regions, successfully bridging and filling in the tiny black holes.
->Erosion step: Shrinks the outer boundaries of the foreground objects back to their original size and shape, without reopening the filled holes.
->Increasing the kernel size (structuring element) from 3×3 to 9×9 during erosion causes a much stronger and more aggressive shrinking effect on white (foreground) objects.

Morphological operations are still widely used because they are simple, fast, and effective for cleaning binary masks.
This is used in:
Cleaning segmentation masks after inference.
Post-processing outputs from U-Net or Mask R-CNN.
Removing small false positives in medical image segmentation.
Preparing images before contour detection or OCR.
Even in modern pipelines, a few lines of morphology can noticeably improve the quality of a segmentation mask.
lesson 11:
A contour is simply:
A continuous curve joining all the boundary points of an object.
Question 1
Why do we usually perform thresholding before contour detection?
->because contours are found by tracking clean boundaries between bright foreground objects and dark background regions.
Question 2
Why is CHAIN_APPROX_SIMPLE preferred over storing every boundary pixel?
->compresses the boundary by storing only key vertices (endpoints) of straight segments, rather than storing every single pixel along the boundary
Question 3
What is the difference between:
Bounding Rectangle
Minimum Enclosing Circle
->A bounding rectangle wraps an object in an axis-aligned box defined by (x, y, width, height), while a minimum enclosing circle wraps it in the smallest possible circle defined by (center, radius).
Question 4:
Suppose your image contains 1,000 tiny dust particles and one large coin.
How could contour area help you detect only the coin?
Explain your reasoning.
->By filtering contours with an area threshold (area > threshold), you can keep the single contour whose area matches the coin's size while instantly discarding all 1,000 tiny particles with near-zero areas.
lesson 12:
A feature is a point in an image that is:
Distinctive
Repeatable
Easy to find again
Question 1
Why are corners generally more useful than edges for tracking objects?
->Corners are more useful than edges because they provide 2D localization, fixing position in both X and Y directions
Question 2
Why doesn't a flat region make a good feature?
->because all neighboring pixels have nearly identical brightness, offering zero contrast or unique geometric structure to track across frames.
Question 3
ORB uses FAST internally.
Why do you think ORB became much more popular than FAST alone?
->ORB became much more popular because FAST alone lacks scale and rotation invariance, whereas ORB makes FAST keypoints invariant to both while adding a fast binary descriptor (BRIEF).
4.Suppose you take a photo of the same building from two different angles.
How could ORB help determine that both images contain the same building, even though the viewpoints differ?
->ORB pairs scale- and rotation-invariant keypoint detection with a fast binary descriptor, allowing a computer to match distinct architectural features across different viewpoints
**
This lesson introduces two concepts that remain fundamental in modern CV:
Keypoints —> locations in an image that are informative.
Descriptors —> numerical representations of those locations.
Deep learning models don't use ORB descriptors directly, but they learn feature representations for image patches that serve a similar purpose. When we reach CNN feature maps and Vision Transformers, you'll see that they're solving the same high-level problem—representing visual information in a way that can be compared and recognized—using learned features instead of handcrafted ones.