lesson1:building the Training Pipeline
Question 1
Why do we use transforms.ToTensor() instead of feeding a PIL image directly to the model?
A PyTorch model expects inputs as multidimensional array tensors containing floating-point numbers.
transforms.ToTensor() converts a PIL image into a model-ready PyTorch Tensor through two critical operations:
Question 2
Why is the image shape (3, 32, 32) instead of (32, 32, 3)?
(3, 32, 32) follows PyTorch's required Channels-First layout (C, H, W):3 = Color Channels (Red, Green, Blue)32 = Height (pixels)32 = Width (pixels)Standard image processing libraries (like OpenCV, PIL, and Matplotlib) use Channels-Last (H, W, C). However, deep learning frameworks like PyTorch use Channels-First (C, H, W) because modern GPUs process matrix operations and convolution kernels far more efficiently when channel data is contiguous in memory.
Question 3
Why do we use:
image.permute(1, 2, 0)
before displaying the image with Matplotlib?
Matplotlib's plt.imshow() expects images in Channels-Last layout: (H, W, C).
Since a PyTorch tensor has shape (C, H, W) at positions (0, 1, 2):Index 0 = Channels (C)Index 1 = Height (H)Index 2 = Width (W)
Running image.permute(1, 2, 0) moves:Dimension 1 (H) to the 1st positionDimension 2 (W) to the 2nd positionDimension 0 (C) to the 3rd positionThis transforms the shape from (3, 32, 32) back to (32, 32, 3) so Matplotlib can render the colors correctly without throwing a shape error.
Modern GPUs are designed to process many samples in parallel, not one by one.
That's why PyTorch introduces the DataLoader.
batch_size=64->64 images at once 
instead of 
Image
 ↓
Model
we have
64 Images
     ↓
Model


Question 1
What is the difference between a Dataset and a DataLoader?
Dataset: Stores individual data samples and their corresponding labels (e.g., retrieving sample o via dataset[i]).
DataLoader: Wraps a Dataset into an iterable that manages batching, shuffling, and parallel multi-process data loading for training.
Question 2
Why is training with mini-batches generally better than processing one image at a time?
GPU Acceleration (Parallelism): GPUs process matrix operations across multiple images simultaneously far more efficiently than sequentially executing single images.
Stable Gradients: Single-image updates create noisy, erratic gradient vectors. Mini-batches compute the average gradient, smoothing optimization updates toward the minimum.
Question 3
After executing:
images, labels = next(iter(train_loader))
what do images and labels contain, and why does images.shape have four dimensions?
images: A tensor containing a single batch of image data scaled and formatted for the network.
labels: A 1D tensor of class target indices corresponding to each image in the batch.
Why 4 Dimensions?
PyTorch formats batch tensors as (B, C, H, W):B = Batch size (number of images)C = Color channels (e.g., 3 for RGB)$H$ = Image height$W$ = Image width
Question 4
Why does shuffle=True matter during training, but we usually set shuffle=False for validation and test loaders?
Training (shuffle=True): Randomizes sequence per epoch to break mini-batch correlations and prevent the model from learning order-dependent bias.
Validation/Testing (shuffle=False): Keeps evaluation deterministic for consistent benchmarks while avoiding unnecessary computation.
Challenge Question (Interview Level)
Suppose you have:
48,000 training images
batch_size = 96
How many batches are there in one epoch?
500
Will the last batch be full or smaller?
Because 48000 is perfectly divisible by 96 (remainder = 0), the final batch contains a complete set of 96 samples.
If you set drop_last=True, what changes? 
If drop_last=True: Nothing changes in this specific scenario.drop_last=True discards incomplete final batches. Since the last batch is already full (96 items), zero samples are dropped, and the total batch count remains $500$.