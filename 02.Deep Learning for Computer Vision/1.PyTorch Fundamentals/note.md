Lesson1:
1. Why Not Just Use NumPy?
Because NumPy has three major limitations
Limitation 1 — No GPU Support
Limitation 2 — No Automatic Gradients:
Suppose a network predicts:Dog
but the answer is:Cat
The network needs to learn.That requires computing derivatives (gradients) of millions of parameters.
Doing that manually is impossible.PyTorch computes them automatically.This is called Autograd.
Limitation 3 — Deep Learning Operations
PyTorch already provides optimized implementations for:
Convolution
Pooling
Backpropagation
Optimizers
Neural network layers
GPU execution
instead of writing them yourself.
2. What is a Tensor?
A tensor is simply a multidimensional array.
3. What is a Batch?
Suppose you have
1 image
Shape=(3 × 224 × 224)
Now suppose you load
32 images
Shape becomes=(32 × 3 × 224 × 224)
The first dimension is the batch size.
Neural networks process many images together because GPUs are optimized for parallel computation.
1.Why does PyTorch use tensors instead of NumPy arrays?
->discussed above.
2.What is the difference between (224,224,3) and (3,224,224)?
->Both shapes represent an RGB image with a resolution of 224*224 pixels and 3 color channels (Red, Green, Blue), but they order the dimensions differently:
HWC Format: (Height, Width, Channels). This is the standard layout used by general image processing libraries like OpenCV, Matplotlib, and PIL.
CHW Format: (Channels, Height, Width). This is the layout required by PyTorch.
3.Why do deep learning models use batches instead of processing one image at a time?
->Processing images in batches (e.g., 32 or 64 at a time) rather than one by one offers three key benefits:

Hardware Parallelism: Modern GPUs contain thousands of small computing cores. Feeding a single image leaves most cores idle. Processing a batch saturates GPU memory bandwidth and keeps compute cores fully utilized.

Vectorized Computations: Matrix multiplication over a 4D batch tensor is vastly faster than running a Python for loop over individual 3D image tensors.

Stable Gradient Estimation: Updating model weights based on a single image leads to noisy, unstable gradient steps. A batch averages the loss across multiple examples, providing smoother convergence during Stochastic Gradient Descent (SGD).
4.If a tensor has shape:
64 × 3 × 224 × 224
what does each dimension represent?
->This represents a standard 4D PyTorch image batch tensor following the NCHW convention:
Lesson2:
Challenge Questions

1.What is the difference between img[0] and img[:,0,0]?
->img[0] selects the first channel (the Red channel). It keeps all height and width dimensions, returning a 2D tensor of shape (224, 224).
->img[:, 0, 0] selects a single pixel at row 0, column 0 across all 3 color channels. It returns a 1D tensor of shape (3,) containing [Red, Green, Blue] values for that top-left pixel.
2.Why is broadcasting faster than writing nested for loops?
->Broadcasting is drastically faster for two primary reasons:
1.C / CUDA C++ Execution (Vectorization)
2.Zero Memory Overhead
3.If an image tensor has shape (3,224,224), what is the shape of img[1]? Why?
->Shape: torch.Size([224, 224])
->(Index 0 = Red, Index 1 = Green, Index 2 = Blue). Indexing img[1] fixes the channel dimension to the Green channel while keeping all remaining spatial dimensions (Height, Width) intact, leaving a 2D matrix representing the Green intensity map.
4.Why do we use torch.clamp() after increasing brightness?
->torch.clamp(img, min=0.0, max=1.0) hard-caps all values, ensuring pixels stay safely within valid numerical limits without affecting relative intensity relationships below 1.0.