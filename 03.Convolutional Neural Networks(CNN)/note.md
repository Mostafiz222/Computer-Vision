Why CNN?
-Problem 1 — Too Many Parameters
Large models mean:
More memory
Slower training
More overfitting
More data required
MLPs do not scale well with image size.
Problem2 - Flattening Destroys Spatial Information
Original:
1 2 3
4 5 6
7 8 9
Flattened:
1 2 3 4 5 6 7 8 9
The model no longer knows:
which pixels are neighbors
where edges are
where corners are
It only sees a long list of numbers.
Why is this bad?
Images are not random numbers.
They contain:
edges
textures
corners
shapes
objects
These depend on neighboring pixels.
CNNs preserve these local relationships.
Key Idea of CNNs:Instead of looking at the whole image at once,CNNs look at small regions.
Imagine reading a book.
You don't look at every page simultaneously.
You read:
Word
↓
Sentence
↓
Paragraph
↓
Chapter
CNN works exactly same:
Pixels
↓
Edges
↓
Textures
↓
Shapes
↓
Objects
##Why CNNs Work Better
A CNN:
looks at local regions
reuses the same filter across the image
preserves spatial information
needs far fewer parameters
generalizes better to real images

| MLP                       | CNN                       |
| ------------------------- | ------------------------- |
| Flattens image            | Keeps image structure     |
| Huge number of parameters | Parameter efficient       |
| No spatial awareness      | Learns spatial patterns   |
| Poor scalability          | Excellent scalability     |
| Good for simple data      | Standard for vision tasks |


##What is Convolution?
->A convolution is an operation that extracts local features from an image.
Instead of looking at the whole image, we look at a small window.

What is a Filter (Kernel)?
A filter (also called a kernel) is a small matrix of learnable numbers.
Example:
3×3 Filter
-1  -1  -1
-1   8  -1
-1  -1  -1
This particular filter highlights edges.
A CNN does not use hand-designed filters. During training, it learns the filter values automatically.
How Convolution Works?
Suppose we have a small image.
Image
1 2 3
4 5 6
7 8 9
Filter
1 0
0 1
Place the filter on the first 2×2 region.
1 2
4 5
Multiply element-wise:
1×1 = 1
2×0 = 0
4×0 = 0
5×1 = 5
Add them:
1 + 0 + 0 + 5 = 6
The first output value is:6
The same filter is reused across the entire image.
This is called parameter sharing.
Instead of learning millions of weights like an MLP, the CNN learns one small filter and applies it everywhere.
#Feature Map
The output of a convolution is called a feature map.
Different filters learn different things.
Example:
Filter 1
↓
Edges
Filter 2
↓
Corners
Filter 3
↓
Textures
Filter 4
↓
Curves
Later layers combine these into:
Eye
↓
Face
↓
Cat

##Why Small Filters?
Because:
fewer parameters
faster
deeper networks
better generalization
why did output become 28*28 ti 26*26?

Filter:3×3
Without padding:
Output size:
28 - 3 + 1=26
So:26×26
##Padding
Padding means adding extra pixels around the image.

#Stride
Stride controls how far the filter moves each time.
Stride reduces image size while extracting features.

Lesson 2:
What is Pooling?
Pooling summarizes a small region into one value.
Example:
Input:
4 2
1 8
Max Pooling
Take the maximum.
Result:8
Another example:
1 3
6 2
Result:6
The strongest feature survives.
3. Max Pooling
This is the most common pooling method.
Example:
Input
1 3 2 1
5 6 1 0
2 4 8 3
1 2 7 5
Apply:
2×2
pooling.
First block:
1 3
5 6
Maximum:6
Second block:
2 1
1 0
Maximum:2
Continue
Final output:
6 2
4 8
Notice:
4×4
↓
2×2
The spatial size becomes half.
Max pooling helps the network remain less sensitive to such small shifts.
This improves robustness
#Average Pooling
Instead of the maximum,Take the average.

##Complete CNN architecture:
Input
↓
Conv
↓
ReLU
↓
Pool
↓
Conv
↓
ReLU
↓
Pool
↓
Flatten
↓
Linear
↓
Output
Visual pipeline:
28×28×1(image)
↓
Conv(16)
↓
28×28×16
↓
Pool
↓
14×14×16
↓
Conv(32)
↓
14×14×32
↓
Pool
↓
7×7×32
↓
Flatten
↓
1568
↓
Linear
↓
128
↓
Linear
↓
10 Classes
Lesson 3:training CNN:

##Data augmentation:
Instead of always showing the exact same image, we randomly modify it during training.
Example:
Original:🐱
Random Flip:🙃
Random Crop:
Zoomed
Random Rotation:
Tilted:The label is still

typical augmentation:
transforms.RandomHorizontalFlip()
transforms.RandomCrop()
transforms.RandomRotation()

Transfer Learning:
Instead of training a CNN from scratch, we start with a model that has already been trained on a huge dataset.
Imagine hiring two people.
Person A
Never seen an image before.
Must learn everything.
Person B
Has already looked at 14 million images.
Only needs to learn your new task.
Who learns faster?
Obviously Person B.
That's transfer learning.