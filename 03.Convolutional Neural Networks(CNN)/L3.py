#trasnfer learning:(short)
#import model resnet18
from torchvision.models import resnet18
model = resnet18(weights="DEFAULT")
print(model)
#freeze:freeze all pretrained parameter
for param in model.parameters():
    param.requires_grad = False
import torch.nn as nn
#replace classifier
model.fc = nn.Linear(
    model.fc.in_features,#input 
    10#output
)
print(model.fc)