from transformers import ViTForImageClassification
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

print(model.config)