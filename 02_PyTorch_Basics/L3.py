# Task 1
import torch
x = torch.tensor(3.0, requires_grad=True)
# 2. Forward pass: y = x^2
y = x ** 2
# 3. Backward pass: compute derivative (dy/dx = 2x)
y.backward()
print("x:     ", x)
print("y:     ", y)
print("x.grad:", x.grad)
#task 2 (use autograd)
# 1. Create scalar tensor x = 2.0 with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
# 2. Forward pass: y = 5 * x^3
y = 5 * (x ** 3)
# 3. Backward pass: compute dy/dx
#->15(x)^2)
y.backward()
print("x:     ", x)
print("y:     ", y)
print("x.grad:", x.grad)
#task3:
import torch

# 1. Create tensors with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 2. Forward pass: y = wx + b
y = w * x + b
#x.grad,dy/dx=w
#w.grad,dy/dw=x
#b.grad,dy/db=1
# 3. Backward pass: compute derivatives
y.backward()

# 4. Print outputs
print("y:     ", y.item())
print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("b.grad:", b.grad)

#Task 4 — A Tiny "Neuron"
# 1. Inputs and Parameters with gradient tracking
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
# 2. Forward Pass: Prediction y = w*x + b
y = w * x + b
# 3. Target value
target = torch.tensor(10.0)
# 4. Compute Squared Error Loss: L = (y - target)^2
loss = (y - target) ** 2
# 5. Backward Pass: Autograd computes dL/dx, dL/dw, dL/db via chain rule
#dL/db=(dL/dy)*(dy/db)->chain rule.
loss.backward()
# 6. Print Results
print(f"Prediction (y): {y.item():.1f}")
print(f"Target:         {target.item():.1f}")
print(f"Loss (L):       {loss.item():.1f}")
print("--- Gradients ---")
print(f"dL/dx:          {x.grad.item():.1f}")
print(f"dL/dw:          {w.grad.item():.1f}")
print(f"dL/db:          {b.grad.item():.1f}")