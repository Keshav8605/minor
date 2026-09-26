import os
import sys
import subprocess
import json
import time

MAX_RETRIES = 1
TIMEOUT_S = 900  # 15 minutes max per inference

def get_status():
    res = subprocess.run(["python", "scripts/run_controlled_evaluation.py", "--status"], capture_output=True, text=True)
    out = res.stdout
    completed = 0
    total = 40
    next_sample = None
    next_condition = None
    
    for line in out.splitlines():
        if "Completed:" in line:
            parts = line.split(":")[-1].strip().split("/")
            completed = int(parts[0])
            total = int(parts[1])
    
    return completed, total

def skip_current_hung(checkpoint_dir):
    ckpt_path = os.path.join(checkpoint_dir, "evaluation_checkpoint.json")
    jsonl_path = os.path.join(checkpoint_dir, "inference_results.jsonl")
    
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
        
    completed_pairs = ckpt["completed_pairs"]
    sample_order = ckpt["sample_order"]
    
    # Determine what was next
    # Build plan
    plan = []
    for i in range(len(sample_order)):
        plan.append((i, "general"))
        plan.append((i, "cultural"))
        
    completed_set = {(p["sample_index"], p["condition"]) for p in completed_pairs}
    
    next_pair = None
    for p in plan:
        if p not in completed_set:
            next_pair = p
            break
            
    if not next_pair:
        return
        
    s_idx, cond = next_pair
    # Find image details
    img_data = next(item for item in sample_order if item["sample_index"] == s_idx)
    
    dummy_record = {
        "sample_index": s_idx,
        "condition": cond,
        "meme_id": f"sample_{s_idx:03d}_{img_data['image_filename']}",
        "image_filename": img_data['image_filename'],
        "image_hash": img_data['image_hash'],
        "ocr_text": "Skipped",
        "ground_truth": img_data['ground_truth'],
        "prediction": 0,
        "humor_probability": None,
        "non_humor_probability": None,
        "reasoning": "SKIPPED: Model hang/deadlock detected by wrapper timeout.",
        "json_error": "HUNG_PROCESS",
        "timed_out": True,
        "hit_max_tokens": False,
        "generated_token_count": 0,
        "inference_duration_s": TIMEOUT_S,
        "cultural_category": None,
        "cultural_dependency": None,
        "cultural_sources": [],
        "retrieved_context": "Skipped",
        "drishtikon_matches_count": 0,
        "drishtikon_diagnostics": "[]",
        "context_passed_to_qwen": False,
        "prompt_hash": ckpt["config_hashes"]["prompt_version_hash"],
        "model_config_hash": ckpt["config_hashes"]["model_config_hash"],
        "cultural_config_hash": ckpt["config_hashes"]["cultural_config_hash"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }
    
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dummy_record) + "\n")
        
    ckpt["completed_pairs"].append({"sample_index": s_idx, "condition": cond})
    ckpt["completed"] += 1
    ckpt["remaining"] -= 1
    ckpt["timeouts"] += 1
    ckpt["last_completed_sample"] = s_idx
    ckpt["last_completed_condition"] = cond
    
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
        
    print(f"--> SKIPPED hung inference ({s_idx}, {cond}).")

def main():
    while True:
        completed, total = get_status()
        if completed >= total:
            print(f"Evaluation complete! {completed}/{total}")
            break
            
        print(f"--- Running inference for {completed+1}/{total} ---")
        
        try:
            # Run one inference
            subprocess.run(
                ["python", "-u", "scripts/run_controlled_evaluation.py", "--continue", "--run-one"],
                timeout=TIMEOUT_S,
                check=True
            )
        except subprocess.TimeoutExpired:
            print(f"!!! PROCESS HUNG (Timeout after {TIMEOUT_S}s). Killing and skipping.")
            # Find the latest eval dir
            eval_dirs = sorted([d for d in os.listdir("results") if d.startswith("evaluation_")])
            if eval_dirs:
                latest_dir = os.path.join("results", eval_dirs[-1])
                skip_current_hung(latest_dir)
        except subprocess.CalledProcessError as e:
            print(f"!!! PROCESS CRASHED (Exit code {e.returncode}). Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()
