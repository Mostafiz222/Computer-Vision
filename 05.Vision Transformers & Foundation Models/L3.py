#clip
from transformers import CLIPProcessor
from transformers import CLIPModel

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

print(model)

#Dino
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "facebook/dino-vitb16"
)

print(model)

#DinoV2:
from transformers import AutoImageProcessor
from transformers import AutoModel

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

print(model)

