from transformers import ViTForImageClassification
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

print(model.config)

#DeiT (Data-efficient Image Transformer)
from transformers import DeiTForImageClassification

model = DeiTForImageClassification.from_pretrained(
    "facebook/deit-base-patch16-224"
)

print(model.config)

#Lesson 5:Swin Transformer
from transformers import SwinForImageClassification

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224"
)

print(model.config)