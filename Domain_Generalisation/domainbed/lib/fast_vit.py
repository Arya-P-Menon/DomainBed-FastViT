import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class HuggingFaceFastViTBackbone(nn.Module):
    def __init__(self, model_name="timm/fastvit_sa12.apple_dist_in1k", num_classes=5):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(1024, num_classes)# Usually last stage dim

    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # shape: [B, 1024, 7, 7]
        pooled = features.mean(dim=[2, 3])  # global average pooling → [B, 1024]
        logits = self.classifier(pooled)  # [B, num_classes]
        return [logits]