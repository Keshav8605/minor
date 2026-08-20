import torch.nn as nn
from transformers import XLMRobertaModel

class TextEncoder(nn.Module):
    def __init__(self, model_id: str = "xlm-roberta-base"):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_id)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
