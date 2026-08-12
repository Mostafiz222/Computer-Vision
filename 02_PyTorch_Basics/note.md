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