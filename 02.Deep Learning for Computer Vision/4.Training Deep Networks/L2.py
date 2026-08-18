#import
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#dataset
transform = transforms.ToTensor()


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)


test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
#create loader
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


test_loader = DataLoader(
    test_data,
    batch_size=64
)

#Model
class MNIST_Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784,256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256,10)

        )


    def forward(self,x):

        return self.network(x)


#create model
model = MNIST_Model()

#Loss function:
criterion = nn.CrossEntropyLoss()

#optimizer:
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

#training loop
epochs = 5
for epoch in range(epochs):

    model.train()

    total_loss = 0


    for images,labels in train_loader:

        predictions = model(images)
        loss = criterion(
            predictions,
            labels
        )
        optimizer.zero_grad()

        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)
    print(
    "Epoch:",
    epoch+1,
    "Loss:",
    average_loss
    )

#evaluation:
model.eval()
correct = 0
total = 0
with torch.no_grad():

    for images,labels in test_loader:


        outputs = model(images)


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)



accuracy = correct/total
print(
    "Accuracy:",
    accuracy
)

#saving model:
torch.save(
    model.state_dict(),
    "mnist_model.pth"
)

#loading model:
model = MNIST_Model()
model.load_state_dict(
    torch.load(
        "mnist_model.pth"
    )
)
model.eval()