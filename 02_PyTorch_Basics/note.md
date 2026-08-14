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
Lesson3:Autograd & Gradients
The gradient tells us:
If x changes slightly, how much does y change?(dy/dx)(differentiation)by forward pass
it changes weight to minimize loss.let's say,x^2. differentiation of it is 2x. if x=6,it becomes 12.that's gradient.
Pytorch calculate it automatically.that's called autograd.
1. Why requires_grad=True Matters?
By default, PyTorch tensors do not track operations to save memory and processing time. Setting requires_grad=True tells PyTorch to build a computational graph in the background for that tensor.
2. What loss.backward() Actually Does?
->When we call loss.backward(), PyTorch traverses the computational graph in reverse from the loss tensor back to all input tensors that have requires_grad=True.
3. Calculating x.grad for y = x^2 at x = 5Function:
 y = x^2
 Derivative: {dy}{dx} = 2x
 Evaluation at x = 5
 {dy}{dx}=2x = 2(5) = 10
 So, x.grad will be 10.0
 4. What Gradients Ultimately Tell Us in a Neural Network
Gradients tell us direction and magnitude for updating parameters to make better predictions:
Gradients don't directly tell the model what the "correct" parameters are.
They tell us how the loss changes when a parameter changes. The optimizer then uses that information to update the parameters.
Lesson4:
The optimizer updates the model's parameters to reduce the loss.
A common choice is stochastic gradient descent (SGD)
1. Why Linear Layers Alone Collapse to a Single Layer
A linear layer applies a linear transformation of the form:
y = xW + b
If you stack multiple linear layers without non-linear activation functions between them, the composite operation remains purely linear:
Output = ((xW1 + b1)W2 + b2)W3 + b3
Expanding this mathematically yields:
Output= x(W1W2W3) + (b1W2W3 + b2W3 + b3)
If we define a single equivalent weight matrix W{effective} = W_1 W_2 W_3 and a single bias vector 
b_{{effective}} = b_1 W_2 W_3 + b_2 W_3 + b_3, the entire multi-layer architecture reduces to:
{Output} = x W_{{effective}} + b_{{effective}}
No matter how deep you build a network with only linear layers, it can only learn straight-line (linear) relationships in data.
2. Core Neural Network Components Explained
->Weights (W) : Multiplicative parameters that scale input signals. They control the strength and importance of the connection between individual neurons across layers.
->Biases (b): Additive offsets added to weighted sums. They allow the activation threshold to shift horizontally, enabling neurons to activate even when all inputs are zero.
->Activation Functions: Non-linear transformations (like ReLU or Sigmoid) applied after linear layers. They introduce non-linearity, breaking the linear collapse and allowing the network to learn complex, non-linear decision boundaries.
->Loss Functions: Mathematical functions that quantify the error or difference between model predictions and ground-truth targets (e.g., Mean Squared Error or Cross-Entropy)
->Optimizers: Algorithms (like SGD or Adam) that use calculated gradients nabla  to update weights and biases driving the loss down toward zero over training steps.

1. Why Activation Functions?
->They add non-linearity. Without them, stacking 100 linear layers is mathematically identical to using just 1 linear layer, preventing the network from learning complex curves or patterns.
2. Why Wx + b Instead of Wx?
->The weight (W) scales the slope, but the bias (b) shifts the position of the decision boundary. Without bias, the function is forced to pass through the origin (0,0).
3. Forward Pass vs. Backpropagation:
->Forward Pass: Moves data forward to calculate the prediction and loss.
->Backpropagation: Moves error backward to calculate gradients (how much each weight caused the error).
4. Why optimizer.zero_grad()?
->PyTorch accumulates (sums up) gradients by default. You must clear old gradients before loss.backward(), or previous training steps will corrupt the new updates
5. What Happens If You Forget optimizer.step()?
->Gradients are calculated, but the model weights never actually update. The model will fail to learn and its loss won't decrease.
6. Why Subclass nn.Module?
->It unlocks built-in PyTorch features automatically: parameter tracking (model.parameters()), one-line GPU support (model.to('cuda')), saving/loading weights, and toggling training/evaluation modes.