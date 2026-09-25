import hashlib
import torch
import numpy as np
import random
import logging
import json
from src.vlm.inference import VLMInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def main():
    print("=== Determinism Investigation ===")
    
    # 1. Exact setup
    image_path = "data/processed/images/train_6346.jpg"
    ocr_text = "Quarantine Day No.5:\nIndian parents: Beta,\nab Umar hogayi,\nShaadi kab karoge?\nKam se kam 30 saal ka toh time de"
    retrieved_context = "EXTERNAL CULTURAL CONTEXT:\nPROJECT CULTURAL KNOWLEDGE:\n- MARRIAGE/WEDDING: Big fat Indian weddings, arranged marriage matrimonial Biodata, nosy relatives asking 'shaadi kab karoge?', buffet food rush, dowry and extravagant ceremonies.\n- FAMILY: Indian family dynamics often involve strict parenting, comparison with peers (e.g., 'Sharma ji ka beta'), emphasis on respect for elders, taunts about marriage or career, close-knit joint family structures, and sibling banter (bhai/behen)."
    
    # 2. Hash inputs
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_hash = hashlib.sha256(img_bytes).hexdigest()
    ocr_hash = compute_hash(ocr_text)
    ctx_hash = compute_hash(retrieved_context)
    
    print(f"Image Hash: {img_hash}")
    print(f"OCR Hash: {ocr_hash}")
    print(f"Context Hash: {ctx_hash}")
    
    engine = VLMInferenceEngine("configs/model.yaml")
    engine.load()
    
    print(f"Generation Params: {engine.generation_params}")
    
    num_runs = 3
    results = []
    
    for i in range(num_runs):
        set_seed(42)  # Re-seed before every run
        print(f"\n--- Run {i+1}/{num_runs} ---")
        
        from src.vlm.prompts import build_cultural_analysis_prompt
        messages = build_cultural_analysis_prompt(image_path, ocr_text, retrieved_context)
            
        prompt_text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_hash = compute_hash(prompt_text)
        print(f"Prompt Hash: {prompt_hash}")
        
        parsed, humor_prob, non_humor_prob = engine._run_generation(messages)
        
        res = {
            "run": i+1,
            "prompt_hash": prompt_hash,
            "humor_prob": humor_prob,
            "non_humor_prob": non_humor_prob,
            "parsed_humorous": parsed.get("humorous")
        }
        print(json.dumps(res, indent=2))
        results.append(res)
        
    print("\n=== Determinism Summary ===")
    probs = [r['humor_prob'] for r in results]
    if len(set(probs)) == 1:
        print("PASS: Inference is deterministic. Probabilities match perfectly.")
    else:
        print(f"FAIL: Non-determinism detected. Probabilities varied: {probs}")

if __name__ == "__main__":
    main()
