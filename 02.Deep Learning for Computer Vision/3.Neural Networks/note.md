lesson1:
layer = nn.Linear(3, 2)
3 input features
2 output neurons(Hidden layer)
How many parameters does this layer have?
Formula:
Parameters=(in_features * out_features)+out_features
Lesson2:
What is nn.Module?
nn.Module is the base class for all neural networks in PyTorch.
Think of it as a container that organizes everything your model needs:
Layers
Parameters
Forward computation
Saving and loading
CPU/GPU transfer
Training and evaluation modes
Instead of managing many separate layers, you create one model object.
In Python, we create our own model by inheriting from nn.Module.
Every PyTorch model usually contains two methods:
1) __init__()
This is where you define the layers.
self.fc1 = nn.Linear(4, 8)
2) forward()
This defines how data flows through the network.
x = self.fc1(x)
return x

Question 1: Purpose of Inheriting from nn.Module

Inheriting from `nn.Module` registers your Python class with PyTorch's internal architecture engine. It gives your class built-in superpowers:
Question 2: Difference Between `__init__()` and `forward()`

-> __init__() (Constructor / Setup Phase): Executed only once when you instantiate the model object (Model = MyModel()). This is where you define and instantiate all the layers (like nn.Linear, `nn.Conv2d`) and components the network will use.

->forward() (Execution / Data Flow Phase): Executed every time data passes through the model. This is where you define the actual mathematical path, specifying how input tensor `x` flows through the layers defined in `__init__()`.

Question 3: Why Call model(x) Instead of model.forward(x)?

Calling `model(x)` triggers Python's `__call__` method implemented in `nn.Module`.

`__call__` does more than just run `forward(x)`—it handles critical PyTorch internal operations, including:

1. Executing PyTorch **forward hooks** (used for logging, debugging, or extracting intermediate activations).
2. Setting up gradient tracking and autograd metadata.
3. Managing internal state tracking before and after passing data.

Calling `model.forward(x)` directly bypasses all these hooks and internal mechanics, which can break certain PyTorch features or profiling tools.

### Question 4: What Breaks If You Remove `super().__init__()`?

If you omit `super().__init__()`, the base `nn.Module` class will **never be initialized**.

This causes immediate breakage:
1. **Attribute Error:** PyTorch will fail when you try to assign layers (e.g., `self.fc1 = nn.Linear(...)`) because the internal dictionary that tracks modules (`self._modules`) doesn't exist.
2. **Untracked Parameters:** `model.parameters()` will throw an error or return an empty list because the parameter tracking system was never set up.
3. **Broken Methods:** Commands like `model.to('cuda')` or `model.state_dict()` will crash.