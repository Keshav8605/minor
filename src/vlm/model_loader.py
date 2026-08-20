import torch
import warnings
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_id: str):
    """
    Loads the Qwen2.5-VL model and its processor.
    Detects CUDA and falls back to CPU if unavailable.
    """
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        print(f"Loading {model_id} on {device} with {torch_dtype}...")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        warnings.warn("CUDA is not available. Falling back to CPU. Inference will be extremely slow.")
        print(f"Loading {model_id} on {device} with {torch_dtype}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
        
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor, model.device
