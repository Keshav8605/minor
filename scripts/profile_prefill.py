"""
Prefill Latency Profiling Script.

Measures separately:
A. Image preprocessing time
B. OCR time (from dataset)
C. Cultural retrieval time
D. Prompt construction time
E. Processor/tokenization time
F. Model input preparation time
G. First-token/prefill latency
H. Subsequent token generation latency
I. Total generation time
J. Total end-to-end inference time

Also measures:
- Input token counts (text + visual)
- Image resolution / visual token budget
- Cultural context size
- CPU/memory behavior
"""

import os
import sys
import gc
import time
import json
import logging
import psutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.vlm.model_loader import load_model_and_processor
from src.vlm.prompts import build_humor_analysis_prompt, build_cultural_analysis_prompt
from src.cultural.context_builder import CulturalContextBuilder
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prefill_profiler")


def get_memory_info():
    """Get current process memory info."""
    proc = psutil.Process()
    mem = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "process_rss_mb": round(mem.rss / 1024 / 1024, 1),
        "process_vms_mb": round(mem.vms / 1024 / 1024, 1),
        "system_total_mb": round(sys_mem.total / 1024 / 1024, 1),
        "system_available_mb": round(sys_mem.available / 1024 / 1024, 1),
        "system_used_percent": sys_mem.percent,
        "swap_used_mb": round(swap.used / 1024 / 1024, 1),
        "swap_total_mb": round(swap.total / 1024 / 1024, 1),
        "swap_percent": swap.percent,
    }


def profile_single_mode(model, processor, device, messages, mode_name, max_new_tokens, generation_params):
    """Profile a single inference run with detailed timing breakdown."""
    
    results = {"mode": mode_name}
    
    # --- E. Processor/tokenization time ---
    t0 = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t_template = time.time() - t0
    results["chat_template_time"] = round(t_template, 4)
    
    # Measure prompt character count
    results["prompt_char_count"] = len(text)
    
    # Count text tokens (approximate via tokenizer)
    t0 = time.time()
    text_token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    t_tokenize = time.time() - t0
    results["text_token_count"] = len(text_token_ids)
    results["tokenizer_encode_time"] = round(t_tokenize, 4)
    
    # --- A. Image preprocessing (via process_vision_info) ---
    t0 = time.time()
    image_inputs, video_inputs = process_vision_info(messages)
    t_vision = time.time() - t0
    results["vision_processing_time"] = round(t_vision, 4)
    
    if image_inputs:
        for i, img in enumerate(image_inputs):
            if hasattr(img, 'size'):
                results[f"image_{i}_size"] = img.size  # (width, height)
            if hasattr(img, 'shape'):
                results[f"image_{i}_shape"] = list(img.shape)
    
    # --- F. Model input preparation ---
    t0 = time.time()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    t_processor = time.time() - t0
    results["processor_time"] = round(t_processor, 4)
    
    # Measure actual input tensor sizes
    input_ids = inputs.input_ids
    results["total_input_tokens"] = input_ids.shape[1]
    
    # Count image/visual tokens vs text tokens
    # Qwen2.5-VL uses special image tokens; count them
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        # Image placeholder token ID
        image_token_id = None
        for name in ['image_token_id', 'image_token']:
            if hasattr(tokenizer, name):
                image_token_id = getattr(tokenizer, name)
                break
        if image_token_id is None:
            # Try to find from special tokens
            for tok_name, tok_id in tokenizer.get_added_vocab().items():
                if 'image' in tok_name.lower() or 'vision' in tok_name.lower():
                    image_token_id = tok_id
                    break
        
        if image_token_id is not None:
            input_list = input_ids[0].tolist()
            visual_tokens = input_list.count(image_token_id)
            results["visual_token_count"] = visual_tokens
        else:
            # Estimate: total input tokens - text tokens
            results["visual_token_count_estimate"] = input_ids.shape[1] - len(text_token_ids)
    
    # Check pixel_values shape if present
    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
        pv = inputs.pixel_values
        if hasattr(pv, 'shape'):
            results["pixel_values_shape"] = list(pv.shape)
        results["pixel_values_dtype"] = str(pv.dtype)
        results["pixel_values_size_mb"] = round(pv.nelement() * pv.element_size() / 1024 / 1024, 2)
    
    # Check image_grid_thw if present (Qwen2.5-VL specific)
    if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
        results["image_grid_thw"] = inputs.image_grid_thw.tolist()
    
    inputs = inputs.to(device)
    
    results["memory_before_generate"] = get_memory_info()
    
    # --- G+H+I. Generation with first-token measurement ---
    gen_params = dict(generation_params)
    gen_params.pop("temperature", None)  # greedy
    gen_params["return_dict_in_generate"] = True
    gen_params["output_scores"] = True
    gen_params["max_new_tokens"] = max_new_tokens
    
    torch.manual_seed(42)
    gc.collect()
    
    t_gen_start = time.time()
    outputs = model.generate(**inputs, **gen_params)
    t_gen_total = time.time() - t_gen_start
    
    results["memory_after_generate"] = get_memory_info()
    
    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    gen_token_count = len(generated_ids_trimmed[0])
    
    results["total_generation_time"] = round(t_gen_total, 2)
    results["generated_token_count"] = gen_token_count
    results["hit_max_new_tokens"] = gen_token_count >= max_new_tokens
    
    # First-token latency vs per-token latency from scores timestamps
    # scores is a tuple with one tensor per generated token
    scores = outputs.scores
    if scores and len(scores) > 0:
        # We can estimate first-token latency as total_time * (proportion of time for first token)
        # Since we don't have per-token timestamps from generate(), we estimate:
        # prefill_time ≈ total_time - (gen_token_count - 1) * avg_decode_time
        # But without per-token timing, we approximate from token count and total time
        if gen_token_count > 1:
            # Rough estimate: first token takes disproportionately longer
            results["avg_per_token_time"] = round(t_gen_total / gen_token_count, 3)
        else:
            results["avg_per_token_time"] = round(t_gen_total, 3)
    
    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    results["output_text"] = output_text[:1000]
    results["output_char_count"] = len(output_text)
    
    return results


def profile_prefill_only(model, processor, device, messages, mode_name):
    """Profile ONLY the prefill phase without generation (max_new_tokens=1)."""
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    torch.manual_seed(42)
    gc.collect()
    
    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    prefill_time = time.time() - t0
    
    return {
        "mode": mode_name,
        "prefill_only_time": round(prefill_time, 2),
        "input_tokens": inputs.input_ids.shape[1],
    }


def run_profiling():
    image_path = "data/processed/images/train_6346.jpg"
    if not Path(image_path).exists():
        image_path = "Memotion 3/testImages/testImages/train_6346.jpg"
    
    # Get original image dimensions
    img = Image.open(image_path)
    original_size = img.size  # (width, height)
    print(f"Original image dimensions: {original_size}")
    
    ocr_text = "Quarantine Day No.5: Indian parents: Beta, ab Umar hogayi, Shaadi kab karoge? Kam se kam 30 saal ka toh time de"
    
    print("\n=== Step 1: Memory Baseline ===")
    print(json.dumps(get_memory_info(), indent=2))
    
    # Load model
    print("\n=== Step 2: Loading Model ===")
    t0 = time.time()
    model, processor, device = load_model_and_processor("Qwen/Qwen2.5-VL-3B-Instruct")
    model_load_time = time.time() - t0
    print(f"Model load time: {model_load_time:.1f}s")
    print(f"Memory after model load: {json.dumps(get_memory_info(), indent=2)}")
    
    # Check processor config for image processing settings
    print("\n=== Step 3: Processor Configuration ===")
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        config_dict = {}
        for attr in ['size', 'image_mean', 'image_std', 'do_resize', 'do_rescale',
                      'do_normalize', 'do_convert_rgb', 'min_pixels', 'max_pixels',
                      'patch_size', 'merge_size', 'temporal_patch_size']:
            if hasattr(ip, attr):
                val = getattr(ip, attr)
                if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                    config_dict[attr] = val
        print(json.dumps(config_dict, indent=2, default=str))
    
    generation_params = {"do_sample": False}
    
    # --- Cultural Retrieval Profiling ---
    print("\n=== Step 4: Cultural Retrieval Profiling ===")
    t0 = time.time()
    builder = CulturalContextBuilder(
        categories_path="data/cultural/cultural_categories.json",
        knowledge_path="data/cultural/cultural_knowledge.json",
        drishtikon_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
        drishtikon_enabled=True,
        drishtikon_top_k=3,
        min_similarity_threshold=0.15
    )
    retrieval_init_time = time.time() - t0
    print(f"Retrieval init time: {retrieval_init_time:.2f}s")
    
    t0 = time.time()
    cult_context, cult_sources, drish_matches = builder.build_context(ocr_text)
    retrieval_time = time.time() - t0
    print(f"Retrieval query time: {retrieval_time:.4f}s")
    print(f"Sources: {cult_sources}")
    print(f"DRISHTIKON matches: {len(drish_matches)}")
    print(f"Cultural context length: {len(cult_context)} chars")
    print(f"Cultural context:\n---\n{cult_context}\n---")
    
    # Estimate cultural context tokens
    cult_context_tokens = len(processor.tokenizer.encode(cult_context, add_special_tokens=False)) if cult_context else 0
    print(f"Cultural context token count: {cult_context_tokens}")
    
    # --- Prompt Construction ---
    print("\n=== Step 5: Prompt Construction ===")
    t0 = time.time()
    general_msgs = build_humor_analysis_prompt(str(Path(image_path).resolve()))
    t_gen_prompt = time.time() - t0
    print(f"General prompt construction: {t_gen_prompt:.4f}s")
    
    t0 = time.time()
    cultural_msgs = build_cultural_analysis_prompt(
        str(Path(image_path).resolve()), ocr_text=ocr_text, retrieved_context=cult_context
    )
    t_cult_prompt = time.time() - t0
    print(f"Cultural prompt construction: {t_cult_prompt:.4f}s")
    
    # --- Prefill-Only Test (max_new_tokens=1) ---
    print("\n=== Step 6: Prefill-Only Test (max_new_tokens=1) ===")
    
    print("\n--- General Mode Prefill ---")
    gen_prefill = profile_prefill_only(model, processor, device, general_msgs, "general")
    print(json.dumps(gen_prefill, indent=2))
    
    print("\n--- Cultural Mode Prefill ---")
    cult_prefill = profile_prefill_only(model, processor, device, cultural_msgs, "cultural")
    print(json.dumps(cult_prefill, indent=2))
    
    # --- Small Generation Test (max_new_tokens=32) ---
    print("\n=== Step 7: Small Generation Test (max_new_tokens=32) ===")
    
    print("\n--- General Mode (32 tokens) ---")
    gen_small = profile_single_mode(model, processor, device, general_msgs, "general", 32, generation_params)
    print(json.dumps({k: v for k, v in gen_small.items() if k != "output_text"}, indent=2, default=str))
    print(f"Output: {gen_small['output_text'][:300]}")
    
    print("\n--- Cultural Mode (32 tokens) ---")
    cult_small = profile_single_mode(model, processor, device, cultural_msgs, "cultural", 32, generation_params)
    print(json.dumps({k: v for k, v in cult_small.items() if k != "output_text"}, indent=2, default=str))
    print(f"Output: {cult_small['output_text'][:300]}")
    
    # --- Full Generation Test (max_new_tokens=384) ---
    print("\n=== Step 8: Full Generation Test (max_new_tokens=384) ===")
    
    print("\n--- General Mode (384 tokens) ---")
    gen_full = profile_single_mode(model, processor, device, general_msgs, "general", 384, generation_params)
    print(json.dumps({k: v for k, v in gen_full.items() if k != "output_text"}, indent=2, default=str))
    print(f"Output: {gen_full['output_text'][:500]}")
    
    print("\n--- Cultural Mode (384 tokens) ---")
    cult_full = profile_single_mode(model, processor, device, cultural_msgs, "cultural", 384, generation_params)
    print(json.dumps({k: v for k, v in cult_full.items() if k != "output_text"}, indent=2, default=str))
    print(f"Output: {cult_full['output_text'][:500]}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("PREFILL PROFILING REPORT")
    print("=" * 60)
    
    print(f"\nImage: train_6346.jpg")
    print(f"Original dimensions: {original_size}")
    
    print(f"\nGeneral Mode:")
    print(f"  Input tokens: {gen_full.get('total_input_tokens', 'N/A')}")
    print(f"  Text tokens: {gen_full.get('text_token_count', 'N/A')}")
    print(f"  Visual tokens (est): {gen_full.get('visual_token_count_estimate', gen_full.get('visual_token_count', 'N/A'))}")
    if 'pixel_values_shape' in gen_full:
        print(f"  Pixel values shape: {gen_full['pixel_values_shape']}")
    if 'image_grid_thw' in gen_full:
        print(f"  Image grid THW: {gen_full['image_grid_thw']}")
    print(f"  Prefill-only time: {gen_prefill['prefill_only_time']}s")
    print(f"  Full generation time (384): {gen_full['total_generation_time']}s")
    print(f"  Generated tokens: {gen_full['generated_token_count']}")
    print(f"  Avg per-token: {gen_full.get('avg_per_token_time', 'N/A')}s")
    
    print(f"\nCultural-Aware Mode:")
    print(f"  Input tokens: {cult_full.get('total_input_tokens', 'N/A')}")
    print(f"  Text tokens: {cult_full.get('text_token_count', 'N/A')}")
    print(f"  Visual tokens (est): {cult_full.get('visual_token_count_estimate', cult_full.get('visual_token_count', 'N/A'))}")
    print(f"  Cultural context tokens: {cult_context_tokens}")
    if 'pixel_values_shape' in cult_full:
        print(f"  Pixel values shape: {cult_full['pixel_values_shape']}")
    if 'image_grid_thw' in cult_full:
        print(f"  Image grid THW: {cult_full['image_grid_thw']}")
    print(f"  Prefill-only time: {cult_prefill['prefill_only_time']}s")
    print(f"  Full generation time (384): {cult_full['total_generation_time']}s")
    print(f"  Generated tokens: {cult_full['generated_token_count']}")
    print(f"  Avg per-token: {cult_full.get('avg_per_token_time', 'N/A')}s")
    
    print(f"\nMemory:")
    mem_after = cult_full.get('memory_after_generate', {})
    print(f"  Process RSS: {mem_after.get('process_rss_mb', 'N/A')} MB")
    print(f"  Process VMS: {mem_after.get('process_vms_mb', 'N/A')} MB")
    print(f"  System RAM used: {mem_after.get('system_used_percent', 'N/A')}%")
    print(f"  Swap used: {mem_after.get('swap_used_mb', 'N/A')} MB / {mem_after.get('swap_total_mb', 'N/A')} MB")
    
    print(f"\nTimeout Design:")
    print(f"  StoppingCriteria is invoked BETWEEN token steps only.")
    print(f"  It CANNOT interrupt a single long prefill/token computation.")
    print(f"  If prefill takes >600s, the timeout fires on the FIRST callback (after token 1).")
    print(f"  A true hard wall-clock timeout would require threading/multiprocessing.")


if __name__ == "__main__":
    run_profiling()
