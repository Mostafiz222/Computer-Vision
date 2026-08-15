import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(8,3)
)

# Simulated input
x = torch.randn(2, 4)

# Inference
model.eval()

with torch.no_grad():

    logits = model(x)

    probabilities = F.softmax(logits, dim=1)

    predictions = logits.argmax(dim=1)

print("Logits:\n", logits)

print("\nProbabilities:\n", probabilities)

print("\nPredicted Classes:\n", predictions)

confidence = probabilities.max(dim=1)
print(confidence)