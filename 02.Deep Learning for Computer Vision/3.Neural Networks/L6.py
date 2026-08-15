import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset
X = torch.randn(100, 4)
y = torch.randint(0, 2, (100,))

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()

    optimizer.step()

    predictions = outputs.argmax(dim=1)

    accuracy = (predictions == y).float().mean()

    print(
        f"Epoch {epoch+1:2d} | "
        f"Loss: {loss.item():.4f} | "
        f"Accuracy: {accuracy.item()*100:.2f}%"
    )

predictions = outputs.argmax(dim=1)

accuracy = (predictions == y).float().mean()

print(accuracy.item())