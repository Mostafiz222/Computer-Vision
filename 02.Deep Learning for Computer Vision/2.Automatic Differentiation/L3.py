# #Part 1 — Understanding CrossEntropyLoss
# import torch
# import torch.nn as nn
# # Raw model outputs (logits)
# logits = torch.tensor([
#     [2.5, 0.3, -1.2],
#     [0.2, 1.8, 0.5]
# ])

# # Ground-truth class indices
# labels = torch.tensor([0, 1])

# criterion = nn.CrossEntropyLoss()

# loss = criterion(logits, labels)

# print(f"Loss: {loss.item():.4f}")
# #Part 2 — Finding Predictions
# predictions = torch.argmax(logits, dim=1)

# print(predictions)

# #Part 3 — A Mini Training Loop
# torch.manual_seed(42)

# class TinyClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(4, 3)

#     def forward(self, x):
#         return self.fc(x)


# model = TinyClassifier()

# inputs = torch.randn(8, 4)
# labels = torch.randint(0, 3, (8,))

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(5):

#     optimizer.zero_grad()

#     logits = model(inputs)

#     loss = criterion(logits, labels)

#     loss.backward()

#     optimizer.step()

#     print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

# #Part 4 — Inspecting Gradients
# for name, param in model.named_parameters():
#     print(f"\n{name}")
#     print(param.grad)

#Coding Task 1:Run the mini training loop and verify that the loss generally decreases over the 5 epochs.
import torch
import torch.nn as nn

torch.manual_seed(42)

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


model = TinyClassifier()

inputs = torch.randn(8, 4)
labels = torch.randint(0, 3, (8,))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):

    optimizer.zero_grad()

    logits = model(inputs)

    loss = criterion(logits, labels)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

# Coding Task 2:Print:
print(logits.shape)
print(labels.shape)
print(loss.shape)

# Coding Task 3
# Print the gradient shape for every parameter:
for name, param in model.named_parameters():
    print(name, param.grad.shape)

# Coding Task 4
# Comment out:
for epoch in range(5):

    optimizer.zero_grad()

    logits = model(inputs)

    loss = criterion(logits, labels)

    loss.backward()

    #optimizer.step()

    print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

#loss remain same.
# gradients are computed
#but parameters stay identical 

# Coding Task 5 (Debugging)
# Intentionally create a wrong label:

torch.manual_seed(42)

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


model = TinyClassifier()

inputs = torch.randn(8, 4)
labels = torch.randint(0, 5, (8,))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):

    optimizer.zero_grad()

    logits = model(inputs)

    loss = criterion(logits, labels)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")

# Coding Task 2:Print:
print(logits.shape)
print(labels.shape)
print(loss.shape)

#output feature 3 but lables has 5 output feature.that's why it through Target 4 is out of bounds