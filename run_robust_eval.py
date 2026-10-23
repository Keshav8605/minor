import os
import sys
import subprocess
import json
import time

MAX_RETRIES = 1
TIMEOUT_S = 1500  # 25 minutes max per inference (aligned with 1,500s timeout plan)

def get_status(results_dir=None):
    cmd = ["python", "scripts/run_controlled_evaluation.py", "--status"]
    if results_dir:
        cmd.extend(["--results-dir", results_dir])
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    completed = 0
    total = 100
    
    for line in out.splitlines():
        if "Completed:" in line:
            parts = line.split(":")[-1].strip().split("/")
            completed = int(parts[0])
            total = int(parts[1])
    
    return completed, total

def skip_current_hung(checkpoint_dir, config_mode="both"):
    ckpt_path = os.path.join(checkpoint_dir, "evaluation_checkpoint.json")
    jsonl_path = os.path.join(checkpoint_dir, "inference_results.jsonl")
    
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
        
    completed_pairs = ckpt["completed_pairs"]
    sample_order = ckpt["sample_order"]
    
    # Determine what was next
    # Build plan
    plan = []
    if config_mode == "general":
        for i in range(len(sample_order)):
            plan.append((i, "general"))
    elif config_mode == "cultural":
        for i in range(len(sample_order)):
            plan.append((i, "cultural"))
    else:
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
        "prediction": None,  # NEVER default to 0 on failure/timeout
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
        f.flush()
        os.fsync(f.fileno())
        
    ckpt["completed_pairs"].append({"sample_index": s_idx, "condition": cond})
    ckpt["completed"] += 1
    ckpt["remaining"] -= 1
    ckpt["timeouts"] += 1
    ckpt["last_completed_sample"] = s_idx
    ckpt["last_completed_condition"] = cond
    
    # Atomic write for checkpoint
    dirpath = os.path.dirname(ckpt_path)
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp", prefix=".ckpt_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ckpt_path)
        
    print(f"--> SKIPPED hung inference ({s_idx}, {cond}) [recorded prediction=None, timed_out=True].")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust Watchdog Evaluation Runner")
    parser.add_argument("--results-dir", type=str, default=None, help="Evaluation results directory")
    parser.add_argument("--manifest", type=str, default=None, help="Path to sample manifest JSON file")
    parser.add_argument("--config-mode", type=str, default="both", choices=["both", "general", "cultural"], help="Evaluation configuration mode")
    parser.add_argument("--stage1-dir", type=str, default=None, help="Directory containing completed Stage-1 results for Cultural mode")
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        if args.config_mode == "general":
            results_dir = "results/evaluation_heldout_100_config_a"
        elif args.config_mode == "cultural":
            results_dir = "results/evaluation_heldout_100_config_b"

    # Base command for single-inference step
    base_cmd = ["python", "-u", "scripts/run_controlled_evaluation.py", "--continue", "--run-one"]
    if results_dir:
        base_cmd.extend(["--results-dir", results_dir])
    if args.manifest:
        base_cmd.extend(["--manifest", args.manifest])
    if args.config_mode:
        base_cmd.extend(["--config-mode", args.config_mode])
    if args.stage1_dir:
        base_cmd.extend(["--stage1-dir", args.stage1_dir])

    while True:
        completed, total = get_status(results_dir)
        if completed >= total and total > 0:
            print(f"Evaluation complete! {completed}/{total}")
            break
            
        print(f"--- Running inference for {completed+1}/{total} ---")
        
        try:
            subprocess.run(
                base_cmd,
                timeout=TIMEOUT_S,
                check=True
            )
        except subprocess.TimeoutExpired:
            print(f"!!! PROCESS HUNG (Timeout after {TIMEOUT_S}s). Killing and skipping.")
            target_dir = results_dir
            if not target_dir:
                eval_dirs = sorted([d for d in os.listdir("results") if d.startswith("evaluation_")])
                if eval_dirs:
                    target_dir = os.path.join("results", eval_dirs[-1])
            if target_dir:
                skip_current_hung(target_dir, config_mode=args.config_mode)
        except subprocess.CalledProcessError as e:
            print(f"!!! PROCESS CRASHED (Exit code {e.returncode}). Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()
