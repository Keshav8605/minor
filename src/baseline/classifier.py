import torch
import torch.nn as nn
from src.baseline.image_encoder import ImageEncoder
from src.baseline.text_encoder import TextEncoder
from src.baseline.fusion import FeatureFusion

class MultimodalClassifier(nn.Module):
    def __init__(self, image_model_id: str, text_model_id: str, hidden_size: int, dropout: float, mode: str = "multimodal"):
        super().__init__()
        self.mode = mode
        
        self.image_encoder = ImageEncoder(image_model_id) if mode in ["multimodal", "image_only"] else None
        self.text_encoder = TextEncoder(text_model_id) if mode in ["multimodal", "text_only"] else None
        self.fusion = FeatureFusion(mode=mode)
        
        img_dim = self.image_encoder.vit.config.hidden_size if self.image_encoder else 0
        txt_dim = self.text_encoder.roberta.config.hidden_size if self.text_encoder else 0
        
        if mode == "multimodal":
            mlp_in_dim = img_dim + txt_dim
        elif mode == "image_only":
            mlp_in_dim = img_dim
        else:
            mlp_in_dim = txt_dim
            
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        img_feat = None
        txt_feat = None
        
        if self.mode in ["multimodal", "image_only"]:
            img_feat = self.image_encoder(pixel_values)
            
        if self.mode in ["multimodal", "text_only"]:
            txt_feat = self.text_encoder(input_ids, attention_mask)
            
        fused = self.fusion(img_feat, txt_feat)
        logits = self.mlp(fused)
        return logits.squeeze(-1)
