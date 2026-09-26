"""
Controlled Research Evaluation: General VLM vs Cultural-Aware VLM (with DRISHTIKON).

Evaluates the exact same test memes under both conditions:
Condition A (General): Meme Image + OCR -> Qwen2.5-VL
Condition B (Cultural-Aware): Meme Image + OCR + Retrieved Context (Project JSON + DRISHTIKON) -> Qwen2.5-VL

Computes accuracy, precision, recall, F1, confusion matrices, and detailed prediction-change analysis.

CHECKPOINT/RESUME:
- Persists every completed inference immediately to JSONL
- Atomic checkpoint file survives laptop shutdown / server restart
- STOP: create stop_signal.flag to pause after current inference
- CONTINUE: rerun script to resume from last checkpoint
- STATUS: use --status flag to report progress without modifying state

REPRODUCIBILITY:
- Saves exact image filenames + image hashes + ground-truth labels
- Saves prompt version/hash + model config + retrieval config
- Records DRISHTIKON retrieval diagnostics per sample
- Enforces torch.manual_seed(42) before each inference

RESEARCH INTEGRITY:
- This checkpoint system is ONLY execution infrastructure
- It does NOT modify inference logic, prompts, metrics, or methodology
"""

import os
import sys
import json
import hashlib
import logging
import random
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.vlm.inference import VLMInferenceEngine
from src.cultural.context_builder import CulturalContextBuilder
from src.vlm.prompts import build_humor_analysis_prompt, build_cultural_analysis_prompt
from src.evaluation.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation_experiment")


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"


def _compute_prompt_version_hash() -> str:
    """Compute a hash over the prompt templates to detect prompt changes."""
    h = hashlib.sha256()
    # Hash both prompt builder outputs for a canonical input
    general_msgs = build_humor_analysis_prompt("dummy.jpg")
    cultural_msgs = build_cultural_analysis_prompt("dummy.jpg", "dummy ocr", "dummy context")
    h.update(json.dumps(general_msgs, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(cultural_msgs, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_latest_evaluation_dir() -> str:
    """Find the most recent evaluation directory with a checkpoint."""
    results_root = Path("results")
    if not results_root.exists():
        return None
    
    eval_dirs = sorted(
        [d for d in results_root.iterdir() 
         if d.is_dir() and d.name.startswith("evaluation_")
         and (d / "evaluation_checkpoint.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    return str(eval_dirs[0]) if eval_dirs else None


def report_status(results_dir: str = None):
    """Report evaluation status without modifying any state."""
    if results_dir is None:
        results_dir = _find_latest_evaluation_dir()
    
    if results_dir is None:
        print("No evaluation checkpoint found.")
        return
    
    mgr = CheckpointManager(results_dir)
    status = mgr.get_status()
    
    print("\n================== EVALUATION STATUS ==================")
    print(f"  Evaluation directory: {status['evaluation_dir']}")
    print(f"  Checkpoint exists:    {status['checkpoint_exists']}")
    print(f"  Status:               {status['status']}")
    print(f"  Completed:            {status['completed']}/{status['total']}")
    print(f"  Remaining:            {status['remaining']}/{status['total']}")
    print(f"  General:              {status['general_completed']}/{status['general_total']}")
    print(f"  Cultural-Aware:       {status['cultural_completed']}/{status['cultural_total']}")
    print(f"  Last completed sample:    {status['last_completed_sample']}")
    print(f"  Last completed condition: {status['last_completed_condition']}")
    print(f"  Timeouts:             {status['timeouts']}")
    print(f"  JSON failures:        {status['json_failures']}")
    print(f"  Stop requested:       {status['stop_requested']}")
    if status['start_time']:
        elapsed = time.time() - status['start_time']
        print(f"  Elapsed time:         {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print("========================================================\n")
    return status


def run_controlled_experiment(
    test_csv_path: str = "data/processed/test.csv",
    image_dir: str = "data/processed/images",
    sample_size: int = 10,
    results_dir: str = None,
    seed: int = 42,
    resume: bool = False,
    run_one: bool = False,
    infer_fn=None,  # Injectable inference function for testing
    context_fn=None,  # Injectable context function for testing
):
    """
    Run the controlled evaluation with persistent checkpoint/resume.
    
    Args:
        test_csv_path: Path to test CSV
        image_dir: Path to image directory
        sample_size: Number of memes (half humorous, half non-humorous)
        results_dir: Output directory (auto-generated if None)
        seed: Random seed for reproducibility
        resume: If True, attempt to resume from existing checkpoint
        infer_fn: Optional mock inference function for testing
        context_fn: Optional mock context function for testing
    """
    
    # ─── Determine results directory ─────────────────────────────────────────
    if resume and results_dir is None:
        results_dir = _find_latest_evaluation_dir()
        if results_dir is None:
            logger.error("No existing evaluation checkpoint found to resume.")
            return None
        logger.info("Resuming from: %s", results_dir)
    
    if results_dir is None:
        results_dir = f"results/evaluation_{int(time.time())}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # ─── Initialize checkpoint manager ───────────────────────────────────────
    mgr = CheckpointManager(results_dir)
    
    # Clear any leftover stop signal when starting fresh or continuing
    mgr.clear_stop_signal()
    
    # ─── Load or create sample selection ─────────────────────────────────────
    existing_checkpoint = mgr.load_checkpoint()
    
    if resume and existing_checkpoint:
        # Restore sample order from checkpoint
        sample_order = existing_checkpoint["sample_order"]
        logger.info("Restored sample order from checkpoint (%d samples)", len(sample_order))
        
        # Rebuild eval_df from sample_order
        df = pd.read_csv(test_csv_path)
        eval_rows = []
        for entry in sample_order:
            matching = df[df["image_filename"] == entry["image_filename"]]
            if len(matching) > 0:
                eval_rows.append(matching.iloc[0])
        eval_df = pd.DataFrame(eval_rows).reset_index(drop=True)
        
        sample_pos_count = sum(1 for s in sample_order if s["ground_truth"] == 1)
        sample_neg_count = sum(1 for s in sample_order if s["ground_truth"] == 0)
        
        # Verify config consistency
        config_hashes = existing_checkpoint.get("config_hashes", {})
        current_prompt_hash = _compute_prompt_version_hash()
        if config_hashes.get("prompt_version_hash") != current_prompt_hash:
            logger.warning(
                "PROMPT HASH CHANGED: checkpoint=%s current=%s",
                config_hashes.get("prompt_version_hash"), current_prompt_hash
            )
        
        start_time = existing_checkpoint.get("start_time", time.time())
    else:
        # Fresh evaluation: select samples
        df = pd.read_csv(test_csv_path)
        humorous_samples = df[df["is_humorous"] == 1]
        non_humorous_samples = df[df["is_humorous"] == 0]
        
        n_pos = sample_size // 2
        n_neg = sample_size - n_pos
        
        sample_pos = humorous_samples.sample(n=min(n_pos, len(humorous_samples)), random_state=seed)
        sample_neg = non_humorous_samples.sample(n=min(n_neg, len(non_humorous_samples)), random_state=seed)
        eval_df = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        sample_pos_count = len(sample_pos)
        sample_neg_count = len(sample_neg)
        start_time = time.time()
    
    total_samples = len(eval_df)
    total_inferences = total_samples * 2  # General + Cultural per sample
    
    logger.info("Selected %d evaluation samples (%d humorous, %d non-humorous)",
                total_samples, sample_pos_count, sample_neg_count)
    
    # ─── Build sample order for checkpoint persistence ───────────────────────
    sample_order = []
    for idx, row in eval_df.iterrows():
        img_filename = row["image_filename"]
        img_path = Path(image_dir) / img_filename
        if not img_path.exists():
            img_path = Path("Memotion 3/testImages/testImages") / img_filename
        sample_order.append({
            "sample_index": idx,
            "image_filename": img_filename,
            "ground_truth": int(row["is_humorous"]),
            "image_hash": _compute_file_hash(str(img_path))[:16],
        })
    
    # ─── Initialize VLM Engine & Cultural Context Builder ────────────────────
    use_mock = infer_fn is not None
    
    if not use_mock:
        engine = VLMInferenceEngine("configs/model.yaml")
        engine.load()
        builder = CulturalContextBuilder(
            categories_path="data/cultural/cultural_categories.json",
            knowledge_path="data/cultural/cultural_knowledge.json",
            drishtikon_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
            drishtikon_enabled=True,
            drishtikon_top_k=3,
            min_similarity_threshold=0.15
        )
    
    # ─── Compute reproducibility metadata ────────────────────────────────────
    prompt_version_hash = _compute_prompt_version_hash()
    model_config_hash = _compute_file_hash("configs/model.yaml")
    cultural_config_hash = _compute_file_hash("configs/cultural.yaml")
    
    config_hashes = {
        "prompt_version_hash": prompt_version_hash,
        "model_config_hash": model_config_hash[:16],
        "cultural_config_hash": cultural_config_hash[:16],
    }
    
    logger.info("Prompt version hash: %s", prompt_version_hash)
    logger.info("Model config hash: %s", model_config_hash[:16])
    logger.info("Cultural config hash: %s", cultural_config_hash[:16])
    
    # ─── Determine completed pairs ───────────────────────────────────────────
    completed_pairs = mgr.get_completed_pairs()
    completed_pairs_list = sorted(completed_pairs, key=lambda x: (x[0], 0 if x[1] == "general" else 1))
    
    logger.info("Already completed: %d/%d inferences", len(completed_pairs), total_inferences)
    
    # ─── Tracking counters ───────────────────────────────────────────────────
    timeouts = 0
    json_failures = 0
    
    # Count from existing results
    existing_results = mgr.load_results()
    for r in existing_results:
        if r.get("timed_out"):
            timeouts += 1
        if r.get("json_error") is not None:
            json_failures += 1
    
    # ─── Build inference plan: ordered list of (sample_index, condition) ──────
    inference_plan = []
    for idx in range(total_samples):
        inference_plan.append((idx, "general"))
        inference_plan.append((idx, "cultural"))
    
    # ─── Execute inferences ──────────────────────────────────────────────────
    for sample_idx, condition in inference_plan:
        # Check stop signal BEFORE starting inference
        if mgr.is_stop_requested():
            logger.info("STOP signal detected. Pausing evaluation.")
            mgr.save_checkpoint(
                evaluation_id=os.path.basename(results_dir),
                total_inferences=total_inferences,
                completed_pairs=completed_pairs_list,
                sample_order=sample_order,
                config_hashes=config_hashes,
                status="paused",
                timeouts=timeouts,
                json_failures=json_failures,
                start_time=start_time,
            )
            print(f"\nEvaluation PAUSED at {len(completed_pairs)}/{total_inferences}.")
            print(f"To resume, run with --continue flag.")
            return None
        
        # Skip already completed pairs
        if (sample_idx, condition) in completed_pairs:
            continue
        
        row = eval_df.iloc[sample_idx]
        meme_id = f"sample_{sample_idx:03d}_{row['image_filename']}"
        img_filename = row["image_filename"]
        img_path = Path(image_dir) / img_filename
        if not img_path.exists():
            img_path = Path("Memotion 3/testImages/testImages") / img_filename
        
        gt = int(row["is_humorous"])
        ocr_text = str(row.get("ocr", ""))
        image_hash = _compute_file_hash(str(img_path))
        
        inference_num = len(completed_pairs) + 1
        logger.info(
            "[%d/%d] Inference: %s | Condition: %s (GT=%d)",
            inference_num, total_inferences, meme_id, condition, gt
        )
        
        inference_start = time.time()
        
        if condition == "general":
            # ─── General Mode ────────────────────────────────────────────
            _set_seed(seed)
            
            if use_mock:
                res = infer_fn(str(img_path), "general", ocr_text, "", [])
            else:
                res = engine.infer(
                    str(img_path),
                    mode="general",
                    ocr_text=ocr_text,
                    retrieved_context="",
                    cultural_sources=[]
                )
            
            inference_duration = time.time() - inference_start
            
            result_record = {
                "sample_index": sample_idx,
                "condition": "general",
                "meme_id": meme_id,
                "image_filename": img_filename,
                "image_hash": image_hash[:16],
                "ocr_text": ocr_text,
                "ground_truth": gt,
                "prediction": 1 if res.get("humorous") else 0,
                "humor_probability": res.get("humor_probability"),
                "non_humor_probability": res.get("non_humor_probability"),
                "reasoning": res.get("reason", ""),
                "json_error": res.get("error"),
                "timed_out": res.get("timed_out", False),
                "hit_max_tokens": res.get("hit_max_new_tokens", False),
                "generated_token_count": res.get("generated_token_count"),
                "inference_duration_s": round(inference_duration, 2),
                "cultural_category": None,
                "cultural_dependency": None,
                "cultural_sources": [],
                "retrieved_context": "",
                "drishtikon_matches_count": 0,
                "drishtikon_diagnostics": "[]",
                "context_passed_to_qwen": False,
                "prompt_hash": config_hashes["prompt_version_hash"],
                "model_config_hash": config_hashes["model_config_hash"],
                "cultural_config_hash": config_hashes["cultural_config_hash"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        
        else:
            # ─── Cultural-Aware Mode ─────────────────────────────────────
            if use_mock and context_fn:
                cult_context, cult_sources, drish_matches = context_fn(ocr_text)
            elif use_mock:
                cult_context, cult_sources, drish_matches = "", [], []
            else:
                cult_context, cult_sources, drish_matches = builder.build_context(ocr_text)
            
            # Record DRISHTIKON retrieval diagnostics
            drish_diagnostics = []
            for match in drish_matches:
                drish_diagnostics.append({
                    "query": ocr_text[:100],
                    "attribute": match.get("attribute", ""),
                    "state": match.get("state", ""),
                    "question": match.get("question", "")[:100],
                    "answer": match.get("answer", "")[:100],
                    "similarity_score": match.get("similarity_score", None),
                    "category": match.get("attribute", ""),
                })
            
            _set_seed(seed)
            
            if use_mock:
                res = infer_fn(str(img_path), "cultural", ocr_text, cult_context, cult_sources)
            else:
                res = engine.infer(
                    str(img_path),
                    mode="cultural",
                    ocr_text=ocr_text,
                    retrieved_context=cult_context,
                    cultural_sources=cult_sources
                )
            
            inference_duration = time.time() - inference_start
            
            result_record = {
                "sample_index": sample_idx,
                "condition": "cultural",
                "meme_id": meme_id,
                "image_filename": img_filename,
                "image_hash": image_hash[:16],
                "ocr_text": ocr_text,
                "ground_truth": gt,
                "prediction": 1 if res.get("humorous") else 0,
                "humor_probability": res.get("humor_probability"),
                "non_humor_probability": res.get("non_humor_probability"),
                "reasoning": res.get("reason", ""),
                "json_error": res.get("error"),
                "timed_out": res.get("timed_out", False),
                "hit_max_tokens": res.get("hit_max_new_tokens", False),
                "generated_token_count": res.get("generated_token_count"),
                "inference_duration_s": round(inference_duration, 2),
                "cultural_category": res.get("cultural_category", ""),
                "cultural_dependency": res.get("cultural_dependency", ""),
                "cultural_sources": cult_sources,
                "retrieved_context": cult_context,
                "drishtikon_matches_count": len(drish_matches),
                "drishtikon_diagnostics": json.dumps(drish_diagnostics),
                "context_passed_to_qwen": bool(cult_context and cult_context.strip()),
                "prompt_hash": config_hashes["prompt_version_hash"],
                "model_config_hash": config_hashes["model_config_hash"],
                "cultural_config_hash": config_hashes["cultural_config_hash"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        
        # ─── PERSIST immediately ─────────────────────────────────────────
        mgr.append_result(result_record)
        
        # Update tracking
        if result_record.get("timed_out"):
            timeouts += 1
        if result_record.get("json_error") is not None:
            json_failures += 1
        
        completed_pairs.add((sample_idx, condition))
        completed_pairs_list = sorted(completed_pairs, key=lambda x: (x[0], 0 if x[1] == "general" else 1))
        
        # ─── CHECKPOINT immediately ──────────────────────────────────────
        mgr.save_checkpoint(
            evaluation_id=os.path.basename(results_dir),
            total_inferences=total_inferences,
            completed_pairs=completed_pairs_list,
            sample_order=sample_order,
            config_hashes=config_hashes,
            status="running",
            timeouts=timeouts,
            json_failures=json_failures,
            start_time=start_time,
        )
        
        logger.info(
            "Persisted: (%d, %s) | %d/%d | Duration: %.1fs",
            sample_idx, condition, len(completed_pairs), total_inferences, inference_duration
        )
        
        if run_one:
            logger.info("Stopping after one inference due to --run-one flag.")
            return None
    
    # ─── All inferences complete ─────────────────────────────────────────────
    logger.info("All %d inferences completed!", total_inferences)
    
    mgr.save_checkpoint(
        evaluation_id=os.path.basename(results_dir),
        total_inferences=total_inferences,
        completed_pairs=completed_pairs_list,
        sample_order=sample_order,
        config_hashes=config_hashes,
        status="completed",
        timeouts=timeouts,
        json_failures=json_failures,
        start_time=start_time,
    )
    
    # ─── Generate final outputs from persisted results ───────────────────────
    return _generate_final_outputs(
        results_dir=results_dir,
        mgr=mgr,
        eval_df=eval_df,
        sample_order=sample_order,
        config_hashes=config_hashes,
        sample_pos_count=sample_pos_count,
        sample_neg_count=sample_neg_count,
        seed=seed,
        engine=engine if not use_mock else None,
    )


def _generate_final_outputs(
    results_dir, mgr, eval_df, sample_order, config_hashes,
    sample_pos_count, sample_neg_count, seed, engine=None
):
    """Generate all final CSV/JSON outputs from persisted JSONL results."""
    
    all_results = mgr.load_results()
    
    # ─── Build comparison_results.csv (one row per meme) ─────────────────────
    # Group results by sample_index
    results_by_sample = {}
    for r in all_results:
        idx = r["sample_index"]
        cond = r["condition"]
        if idx not in results_by_sample:
            results_by_sample[idx] = {}
        results_by_sample[idx][cond] = r
    
    records = []
    for idx in range(len(eval_df)):
        gen_r = results_by_sample.get(idx, {}).get("general", {})
        cult_r = results_by_sample.get(idx, {}).get("cultural", {})
        
        gen_pred = gen_r.get("prediction", 0)
        cult_pred = cult_r.get("prediction", 0)
        gen_prob = gen_r.get("humor_probability")
        cult_prob = cult_r.get("humor_probability")
        
        record = {
            "meme_id": gen_r.get("meme_id", cult_r.get("meme_id", "")),
            "image_filename": gen_r.get("image_filename", cult_r.get("image_filename", "")),
            "image_hash": gen_r.get("image_hash", cult_r.get("image_hash", "")),
            "ocr_text": gen_r.get("ocr_text", cult_r.get("ocr_text", "")),
            "ground_truth": gen_r.get("ground_truth", cult_r.get("ground_truth", 0)),
            "general_prediction": gen_pred,
            "general_humor_probability": gen_prob,
            "general_reasoning": gen_r.get("reasoning", ""),
            "general_json_error": gen_r.get("json_error"),
            "general_inference_duration_s": gen_r.get("inference_duration_s"),
            "general_generated_tokens": gen_r.get("generated_token_count"),
            "cultural_prediction": cult_pred,
            "cultural_humor_probability": cult_prob,
            "cultural_category": cult_r.get("cultural_category", ""),
            "cultural_dependency": cult_r.get("cultural_dependency", ""),
            "cultural_sources": cult_r.get("cultural_sources", []),
            "retrieved_context": cult_r.get("retrieved_context", ""),
            "cultural_reasoning": cult_r.get("reasoning", ""),
            "cultural_json_error": cult_r.get("json_error"),
            "cultural_inference_duration_s": cult_r.get("inference_duration_s"),
            "cultural_generated_tokens": cult_r.get("generated_token_count"),
            "drishtikon_matches_count": cult_r.get("drishtikon_matches_count", 0),
            "drishtikon_diagnostics": cult_r.get("drishtikon_diagnostics", "[]"),
            "context_passed_to_qwen": cult_r.get("context_passed_to_qwen", False),
            "prediction_changed": gen_pred != cult_pred,
            "prob_delta": round((cult_prob - gen_prob), 4) if (cult_prob is not None and gen_prob is not None) else None,
        }
        records.append(record)
    
    res_df = pd.DataFrame(records)
    res_df.to_csv(f"{results_dir}/comparison_results.csv", index=False)
    
    # ─── Compute Metrics ─────────────────────────────────────────────────────
    y_true = res_df["ground_truth"].tolist()
    y_gen = res_df["general_prediction"].tolist()
    y_cult = res_df["cultural_prediction"].tolist()
    
    gen_acc = accuracy_score(y_true, y_gen)
    gen_prec = precision_score(y_true, y_gen, zero_division=0)
    gen_rec = recall_score(y_true, y_gen, zero_division=0)
    gen_f1 = f1_score(y_true, y_gen, zero_division=0)
    gen_cm = confusion_matrix(y_true, y_gen).tolist()
    
    cult_acc = accuracy_score(y_true, y_cult)
    cult_prec = precision_score(y_true, y_cult, zero_division=0)
    cult_rec = recall_score(y_true, y_cult, zero_division=0)
    cult_f1 = f1_score(y_true, y_cult, zero_division=0)
    cult_cm = confusion_matrix(y_true, y_cult).tolist()
    
    # ─── Prediction-Change Breakdown ─────────────────────────────────────────
    gen_wrong_cult_correct = int(((res_df["general_prediction"] != res_df["ground_truth"]) & (res_df["cultural_prediction"] == res_df["ground_truth"])).sum())
    gen_correct_cult_wrong = int(((res_df["general_prediction"] == res_df["ground_truth"]) & (res_df["cultural_prediction"] != res_df["ground_truth"])).sum())
    both_correct = int(((res_df["general_prediction"] == res_df["ground_truth"]) & (res_df["cultural_prediction"] == res_df["ground_truth"])).sum())
    both_wrong = int(((res_df["general_prediction"] != res_df["ground_truth"]) & (res_df["cultural_prediction"] != res_df["ground_truth"])).sum())
    
    # ─── JSON parsing success rates ──────────────────────────────────────────
    gen_json_failures = int(res_df["general_json_error"].notna().sum())
    cult_json_failures = int(res_df["cultural_json_error"].notna().sum())
    
    # ─── Timing statistics ───────────────────────────────────────────────────
    gen_durations = [r.get("inference_duration_s", 0) for r in all_results if r.get("condition") == "general"]
    cult_durations = [r.get("inference_duration_s", 0) for r in all_results if r.get("condition") == "cultural"]
    
    max_new_tokens = None
    temperature = None
    do_sample = None
    if engine:
        max_new_tokens = engine.generation_params.get("max_new_tokens")
        temperature = engine.generation_params.get("temperature")
        do_sample = engine.generation_params.get("do_sample")
    
    summary = {
        "evaluation_dataset": "Memotion 3 Test Set",
        "total_samples": len(eval_df),
        "humorous_ground_truth": sample_pos_count,
        "non_humorous_ground_truth": sample_neg_count,
        "reproducibility": {
            "seed": seed,
            "prompt_version_hash": config_hashes["prompt_version_hash"],
            "model_config_hash": config_hashes["model_config_hash"],
            "cultural_config_hash": config_hashes["cultural_config_hash"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        },
        "json_parsing": {
            "general_failures": gen_json_failures,
            "cultural_failures": cult_json_failures,
            "total_inferences": len(eval_df) * 2,
        },
        "timing": {
            "general_mean_s": round(np.mean(gen_durations), 1) if gen_durations else None,
            "general_max_s": round(max(gen_durations), 1) if gen_durations else None,
            "cultural_mean_s": round(np.mean(cult_durations), 1) if cult_durations else None,
            "cultural_max_s": round(max(cult_durations), 1) if cult_durations else None,
            "total_inference_time_s": round(sum(gen_durations) + sum(cult_durations), 1),
        },
        "general_mode_metrics": {
            "accuracy": round(gen_acc, 4),
            "precision": round(gen_prec, 4),
            "recall": round(gen_rec, 4),
            "f1_score": round(gen_f1, 4),
            "confusion_matrix": gen_cm
        },
        "cultural_mode_metrics": {
            "accuracy": round(cult_acc, 4),
            "precision": round(cult_prec, 4),
            "recall": round(cult_rec, 4),
            "f1_score": round(cult_f1, 4),
            "confusion_matrix": cult_cm
        },
        "delta": {
            "accuracy_change": round(cult_acc - gen_acc, 4),
            "f1_change": round(cult_f1 - gen_f1, 4)
        },
        "prediction_changes": {
            "general_wrong_to_cultural_correct": gen_wrong_cult_correct,
            "general_correct_to_cultural_wrong": gen_correct_cult_wrong,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "total_predictions_changed": int(res_df["prediction_changed"].sum())
        }
    }
    
    with open(f"{results_dir}/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    changed_df = res_df[res_df["prediction_changed"] == True]
    changed_df.to_csv(f"{results_dir}/changed_predictions.csv", index=False)
    
    # Save sample manifest for reproducibility
    with open(f"{results_dir}/sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sample_order, f, indent=2)
    
    logger.info("Evaluation Complete. Results saved in %s", results_dir)
    print("\n================== EVALUATION SUMMARY ==================")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Controlled Research Evaluation")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--status", action="store_true",
                        help="Report evaluation status without modifying state")
    parser.add_argument("--stop", action="store_true",
                        help="Create stop signal to pause after current inference")
    parser.add_argument("--run-one", action="store_true",
                        help="Stop immediately after running one inference")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Evaluation results directory")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of memes to evaluate")
    args = parser.parse_args()
    
    if args.status:
        report_status(args.results_dir)
    elif args.stop:
        results_dir = args.results_dir or _find_latest_evaluation_dir()
        if results_dir:
            mgr = CheckpointManager(results_dir)
            mgr.request_stop()
            print(f"Stop signal created in {results_dir}")
        else:
            print("No evaluation directory found.")
    else:
        run_controlled_experiment(
            sample_size=args.sample_size,
            results_dir=args.results_dir,
            resume=args.resume,
            run_one=args.run_one,
        )
