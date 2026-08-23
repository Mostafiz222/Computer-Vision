#SlipLip
from transformers import AutoProcessor
from transformers import AutoModel

processor = AutoProcessor.from_pretrained(
    "google/siglip-base-patch16-224"
)

model = AutoModel.from_pretrained(
    "google/siglip-base-patch16-224"
)

print(model)

#Linear Probing:
backbone.eval()

for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(768, 100)
#LoRa (PEFT)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1
)

model = get_peft_model(model, config)

model.print_trainable_parameters()