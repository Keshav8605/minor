import torch
import warnings
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_id: str):
    """
    Loads the Qwen2.5-VL model and its processor.
    Detects CUDA and falls back to CPU if unavailable.
    Uses float16 on CPU to reduce memory usage by half.
    """
    # Free up any existing memory before loading
    gc.collect()
    
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        print(f"Loading {model_id} on {device} with {torch_dtype}...")
    else:
        device = "cpu"
        torch_dtype = torch.float16
        warnings.warn("CUDA is not available. Falling back to CPU with float16. Inference will be slow.")
        print(f"Loading {model_id} on {device} with {torch_dtype} (half precision to save memory)...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model.to(device)
        
    processor = AutoProcessor.from_pretrained(model_id)
    
    gc.collect()
    return model, processor, model.device

