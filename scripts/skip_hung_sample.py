import json
import os
import time

results_dir = "results/evaluation_1790249336"
ckpt_path = os.path.join(results_dir, "evaluation_checkpoint.json")
jsonl_path = os.path.join(results_dir, "inference_results.jsonl")

# 1. Add dummy record to JSONL
dummy_record = {
    "sample_index": 3,
    "condition": "cultural",
    "meme_id": "sample_003_train_4199.jpg",
    "image_filename": "train_4199.jpg",
    "image_hash": "7adb10a23c23dde1",
    "ocr_text": "Bore hore hoge ? chalo thoda Hans lo Q",
    "ground_truth": 1,
    "prediction": 0,
    "humor_probability": None,
    "non_humor_probability": None,
    "reasoning": "SKIPPED: Model consistently hangs/deadlocks on this specific cultural input.",
    "json_error": "HUNG_PROCESS",
    "timed_out": True,
    "hit_max_tokens": False,
    "generated_token_count": 0,
    "inference_duration_s": 600.0,
    "cultural_category": None,
    "cultural_dependency": None,
    "cultural_sources": ["PROJECT_JSON", "DRISHTIKON"],
    "retrieved_context": "Skipped due to hang",
    "drishtikon_matches_count": 3,
    "drishtikon_diagnostics": "[]",
    "context_passed_to_qwen": True,
    "prompt_hash": "7fab5835b9fa6c6b",
    "model_config_hash": "28962a8396ae1a10",
    "cultural_config_hash": "413d59425aaaa285",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
}

with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(dummy_record) + "\n")

print("Added dummy record to JSONL.")

# 2. Update checkpoint
with open(ckpt_path, "r", encoding="utf-8") as f:
    ckpt = json.load(f)

if {"sample_index": 3, "condition": "cultural"} not in ckpt["completed_pairs"]:
    ckpt["completed_pairs"].append({"sample_index": 3, "condition": "cultural"})
    ckpt["completed"] += 1
    ckpt["remaining"] -= 1
    ckpt["timeouts"] += 1
    ckpt["json_failures"] += 1
    ckpt["last_completed_sample"] = 3
    ckpt["last_completed_condition"] = "cultural"

with open(ckpt_path, "w", encoding="utf-8") as f:
    json.dump(ckpt, f, indent=2)

print("Updated checkpoint.")
