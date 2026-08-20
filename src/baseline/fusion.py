import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, mode: str = "multimodal"):
        super().__init__()
        self.mode = mode
        
    def forward(self, image_features, text_features):
        if self.mode == "image_only":
            return image_features
        elif self.mode == "text_only":
            return text_features
        elif self.mode == "multimodal":
            return torch.cat((image_features, text_features), dim=1)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
