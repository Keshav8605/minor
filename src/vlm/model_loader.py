import os
import torch
import warnings
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_id: str, min_pixels: int = 200704, max_pixels: int = 401408):
    """
    Loads the Qwen2.5-VL model and its processor.
    Detects CUDA and falls back to CPU if unavailable.
    Uses float16 on CPU to reduce memory usage by half.
    Caps max_pixels to optimize vision token count and CPU latency.
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
        # Optimize CPU threads for performance cores
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        warnings.warn(f"CUDA is not available. Running on CPU with float16 ({num_threads} threads).")
        print(f"Loading {model_id} on {device} with {torch_dtype} ({num_threads} threads, max_pixels={max_pixels})...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model.to(device)

    # Initialize processor with capped pixels to prevent 2000+ vision token explosion on CPU
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    gc.collect()
    return model, processor, model.device


