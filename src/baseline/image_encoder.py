import torch.nn as nn
from transformers import ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, model_id: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_id)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]
