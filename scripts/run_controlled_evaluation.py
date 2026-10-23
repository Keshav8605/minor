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
import yaml
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
    max_samples: int = None,
    infer_fn=None,  # Injectable inference function for testing
    context_fn=None,  # Injectable context function for testing
    manifest_path: str = None,  # Optional path to sample manifest JSON
    config_mode: str = "both",  # "general" (Config A), "cultural" (Config B), or "both"
    stage1_dir: str = None,  # Optional directory to load completed Stage-1 results
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
        manifest_path: Optional path to sample manifest JSON
        config_mode: Evaluation configuration mode ("general", "cultural", or "both")
        stage1_dir: Directory containing completed Stage-1 results for Cultural mode
    """
    
    # ─── Determine results directory ─────────────────────────────────────────
    if resume and results_dir is None:
        results_dir = _find_latest_evaluation_dir()
        if results_dir is None:
            logger.error("No existing evaluation checkpoint found to resume.")
            return None
        logger.info("Resuming from: %s", results_dir)
    
    if results_dir is None:
        if config_mode == "general":
            results_dir = "results/evaluation_heldout_100_config_a"
        elif config_mode == "cultural":
            results_dir = "results/evaluation_heldout_100_config_b"
        else:
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
        if manifest_path and os.path.exists(manifest_path):
            logger.info("Loading evaluation samples from manifest: %s", manifest_path)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_doc = json.load(f)
            manifest_samples = manifest_doc.get("samples", [])
            eval_rows = []
            for s in manifest_samples:
                eval_rows.append({
                    "image_filename": s["image_filename"],
                    "is_humorous": s["ground_truth"],
                    "ocr": s.get("ocr_text", ""),
                    "humour": s.get("original_labels", {}).get("humour"),
                    "sarcastic": s.get("original_labels", {}).get("sarcastic"),
                    "offensive": s.get("original_labels", {}).get("offensive"),
                    "motivational": s.get("original_labels", {}).get("motivational"),
                    "stratum_id": s.get("stratum_id"),
                    "stratum_name": s.get("stratum_name"),
                    "ambiguity_status": s.get("ambiguity_status"),
                    "sample_id": s.get("sample_id"),
                })
            eval_df = pd.DataFrame(eval_rows).reset_index(drop=True)
            sample_pos_count = sum(1 for s in manifest_samples if s["ground_truth"] == 1)
            sample_neg_count = sum(1 for s in manifest_samples if s["ground_truth"] == 0)
            start_time = time.time()
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
    if config_mode == "general":
        total_inferences = total_samples
    elif config_mode == "cultural":
        total_inferences = total_samples
    else:
        total_inferences = total_samples * 2  # General + Cultural per sample
    
    logger.info("Selected %d evaluation samples (%d humorous, %d non-humorous) [Mode: %s, Inferences: %d]",
                total_samples, sample_pos_count, sample_neg_count, config_mode, total_inferences)
    
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
        # Load cultural config parameters
        cult_yaml_path = "configs/cultural.yaml"
        cult_thresh = 0.25
        if os.path.exists(cult_yaml_path):
            with open(cult_yaml_path, "r", encoding="utf-8") as f:
                c_yaml = yaml.safe_load(f) or {}
                cult_thresh = float(c_yaml.get("min_similarity_threshold", 0.25))

        builder = CulturalContextBuilder(
            categories_path="data/cultural/cultural_categories.json",
            knowledge_path="data/cultural/cultural_knowledge.json",
            drishtikon_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
            drishtikon_enabled=True,
            drishtikon_top_k=3,
            min_similarity_threshold=cult_thresh
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
    
    # ─── Validate checkpoint configuration matches current config ─────────────
    mgr.validate_configuration(config_hashes)
    
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
    if config_mode == "general":
        for idx in range(total_samples):
            inference_plan.append((idx, "general"))
    elif config_mode == "cultural":
        for idx in range(total_samples):
            inference_plan.append((idx, "cultural"))
    else:
        for idx in range(total_samples):
            inference_plan.append((idx, "general"))
            inference_plan.append((idx, "cultural"))
    
    # ─── Execute inferences ──────────────────────────────────────────────────
    for sample_idx, condition in inference_plan:
        if max_samples is not None and sample_idx >= max_samples:
            logger.info("Reached max_samples limit (%d). Concluding current subset.", max_samples)
            break

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
            
            # Safety: Ensure failed/timed-out inferences NEVER default to class 0
            is_gen_timed_out = bool(res.get("timed_out", False))
            has_gen_error = res.get("error") is not None
            if is_gen_timed_out or has_gen_error:
                gen_pred_val = None
            elif res.get("humorous") is True:
                gen_pred_val = 1
            elif res.get("humorous") is False:
                gen_pred_val = 0
            else:
                gen_pred_val = None

            result_record = {
                "sample_index": sample_idx,
                "condition": "general",
                "meme_id": meme_id,
                "image_filename": img_filename,
                "image_hash": image_hash[:16],
                "ocr_text": ocr_text,
                "ground_truth": gt,
                "prediction": gen_pred_val,
                "humor_probability": res.get("humor_probability") if gen_pred_val is not None else None,
                "non_humor_probability": res.get("non_humor_probability") if gen_pred_val is not None else None,
                "reasoning": res.get("reason", "") or ("Inference timed out" if is_gen_timed_out else "Inference failed"),
                "json_error": res.get("error") or ("TIMEOUT" if is_gen_timed_out else None),
                "timed_out": is_gen_timed_out,
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
            
            # Check if general condition for this sample has already completed to supply as Stage-1
            stage1_result = None
            if not use_mock:
                completed_results = mgr.load_results()
                for prev in completed_results:
                    if prev.get("sample_index") == sample_idx and prev.get("condition") in ("general", None):
                        if prev.get("prediction") is not None and prev.get("prediction") in (0, 1):
                            stage1_result = {
                                "humorous": bool(prev.get("prediction") == 1),
                                "prediction": "Humorous" if prev.get("prediction") == 1 else "Not Humorous",
                                "humor_probability": prev.get("humor_probability"),
                                "non_humor_probability": prev.get("non_humor_probability"),
                                "confidence": max(prev.get("humor_probability") or 0.5, prev.get("non_humor_probability") or 0.5),
                                "reason": prev.get("reasoning", ""),
                                "detected_text": prev.get("ocr_text", ""),
                            }
                        break
                
                # If not found in current directory, check stage1_dir (e.g. Config A output dir)
                if stage1_result is None and stage1_dir and os.path.exists(stage1_dir):
                    s1_jl = os.path.join(stage1_dir, "inference_results.jsonl")
                    if os.path.exists(s1_jl):
                        with open(s1_jl, "r", encoding="utf-8") as s1_f:
                            for s1_line in s1_f:
                                if s1_line.strip():
                                    prev = json.loads(s1_line)
                                    if prev.get("sample_index") == sample_idx:
                                        if prev.get("prediction") is not None and prev.get("prediction") in (0, 1):
                                            stage1_result = {
                                                "humorous": bool(prev.get("prediction") == 1),
                                                "prediction": "Humorous" if prev.get("prediction") == 1 else "Not Humorous",
                                                "humor_probability": prev.get("humor_probability"),
                                                "non_humor_probability": prev.get("non_humor_probability"),
                                                "confidence": max(prev.get("humor_probability") or 0.5, prev.get("non_humor_probability") or 0.5),
                                                "reason": prev.get("reasoning", ""),
                                                "detected_text": prev.get("ocr_text", ""),
                                            }
                                        break

            if use_mock:
                res = infer_fn(str(img_path), "cultural", ocr_text, cult_context, cult_sources)
            else:
                res = engine.infer(
                    str(img_path),
                    mode="cultural",
                    ocr_text=ocr_text,
                    retrieved_context=cult_context,
                    cultural_sources=cult_sources,
                    stage1_result=stage1_result,
                    two_stage=True
                )
            
            inference_duration = time.time() - inference_start
            
            # Safety: Ensure failed/timed-out inferences NEVER default to class 0
            is_cult_timed_out = bool(res.get("timed_out", False))
            has_cult_error = bool(res.get("error") or res.get("stage2_error"))
            fallback_applied = bool(res.get("stage2_fallback_to_stage1", False))
            
            if is_cult_timed_out or (has_cult_error and not fallback_applied):
                cult_pred_val = None
            elif res.get("humorous") is True:
                cult_pred_val = 1
            elif res.get("humorous") is False:
                cult_pred_val = 0
            else:
                cult_pred_val = None

            result_record = {
                "sample_index": sample_idx,
                "condition": "cultural",
                "meme_id": meme_id,
                "image_filename": img_filename,
                "image_hash": image_hash[:16],
                "ocr_text": ocr_text,
                "ground_truth": gt,
                "prediction": cult_pred_val,
                "humor_probability": res.get("humor_probability") if cult_pred_val is not None else None,
                "non_humor_probability": res.get("non_humor_probability") if cult_pred_val is not None else None,
                "reasoning": res.get("reason", "") or ("Inference timed out" if is_cult_timed_out else "Inference failed"),
                "json_error": res.get("error") or res.get("stage2_error") or ("TIMEOUT" if is_cult_timed_out else None),
                "stage2_failed": res.get("stage2_failed", False),
                "stage2_fallback_to_stage1": fallback_applied,
                "timed_out": is_cult_timed_out,
                "hit_max_tokens": res.get("hit_max_new_tokens", False),
                "generated_token_count": res.get("generated_token_count"),
                "inference_duration_s": round(inference_duration, 2),
                "cultural_category": res.get("cultural_category", ""),
                "cultural_dependency": res.get("cultural_dependency", ""),
                "cultural_sources": res.get("cultural_sources", cult_sources) if res.get("context_passed_to_qwen", True) else [],
                "retrieved_context": cult_context if res.get("context_passed_to_qwen", True) else "",
                "drishtikon_matches_count": len(drish_matches),
                "drishtikon_diagnostics": json.dumps(drish_diagnostics),
                "context_passed_to_qwen": res.get("context_passed_to_qwen", bool(cult_context and cult_context.strip())),
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
    
    completed_indices = [idx for idx in sorted(results_by_sample.keys()) if "general" in results_by_sample[idx] and "cultural" in results_by_sample[idx]]
    if not completed_indices:
        completed_indices = sorted(results_by_sample.keys())

    records = []
    for idx in completed_indices:
        gen_r = results_by_sample.get(idx, {}).get("general", {})
        cult_r = results_by_sample.get(idx, {}).get("cultural", {})
        
        gen_pred = gen_r.get("prediction", None)
        cult_pred = cult_r.get("prediction", None)
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
            "general_timed_out": gen_r.get("timed_out", False),
            "general_hit_max_tokens": gen_r.get("hit_max_tokens", False),
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
            "cultural_stage2_failed": cult_r.get("stage2_failed", False),
            "cultural_stage2_fallback_to_stage1": cult_r.get("stage2_fallback_to_stage1", False),
            "cultural_timed_out": cult_r.get("timed_out", False),
            "cultural_hit_max_tokens": cult_r.get("hit_max_tokens", False),
            "cultural_inference_duration_s": cult_r.get("inference_duration_s"),
            "cultural_generated_tokens": cult_r.get("generated_token_count"),
            "drishtikon_matches_count": cult_r.get("drishtikon_matches_count", 0),
            "drishtikon_diagnostics": cult_r.get("drishtikon_diagnostics", "[]"),
            "context_passed_to_qwen": cult_r.get("context_passed_to_qwen", False),
            "prediction_changed": (gen_pred != cult_pred) if (gen_pred is not None and cult_pred is not None) else False,
            "prob_delta": round((cult_prob - gen_prob), 4) if (cult_prob is not None and gen_prob is not None) else None,
        }
        records.append(record)
    
    res_df = pd.DataFrame(records)
    res_df.to_csv(f"{results_dir}/comparison_results.csv", index=False)
    
    # ─── Evaluated Sample and Class Counts ────────────────────────────────────
    evaluated_samples_count = len(res_df)
    manifest_samples_count = len(eval_df)
    eval_pos_count = int((res_df["ground_truth"] == 1).sum()) if len(res_df) > 0 else 0
    eval_neg_count = int((res_df["ground_truth"] == 0).sum()) if len(res_df) > 0 else 0

    # ─── Compute Metrics (Exclude Failed, Timed Out, or Corrupt Inferences) ──
    gen_timed_out = res_df["general_timed_out"].fillna(False) if "general_timed_out" in res_df.columns else pd.Series(False, index=res_df.index)
    cult_timed_out = res_df["cultural_timed_out"].fillna(False) if "cultural_timed_out" in res_df.columns else pd.Series(False, index=res_df.index)

    gen_valid_mask = res_df["general_prediction"].notna() & res_df["general_prediction"].isin([0, 1]) & (~gen_timed_out) & res_df["general_json_error"].isna()
    cult_valid_mask = res_df["cultural_prediction"].notna() & res_df["cultural_prediction"].isin([0, 1]) & (~cult_timed_out) & res_df["cultural_json_error"].isna()
    
    y_true_gen = res_df.loc[gen_valid_mask, "ground_truth"].tolist()
    y_gen = res_df.loc[gen_valid_mask, "general_prediction"].astype(int).tolist()
    
    y_true_cult = res_df.loc[cult_valid_mask, "ground_truth"].tolist()
    y_cult = res_df.loc[cult_valid_mask, "cultural_prediction"].astype(int).tolist()
    
    if len(y_gen) > 0:
        gen_acc = accuracy_score(y_true_gen, y_gen)
        gen_prec = precision_score(y_true_gen, y_gen, pos_label=1, zero_division=0)
        gen_rec = recall_score(y_true_gen, y_gen, pos_label=1, zero_division=0)
        gen_f1 = f1_score(y_true_gen, y_gen, pos_label=1, zero_division=0)
        gen_nonhumor_prec = precision_score(y_true_gen, y_gen, pos_label=0, zero_division=0)
        gen_nonhumor_rec = recall_score(y_true_gen, y_gen, pos_label=0, zero_division=0)
        gen_nonhumor_f1 = f1_score(y_true_gen, y_gen, pos_label=0, zero_division=0)
        gen_cm = confusion_matrix(y_true_gen, y_gen, labels=[0, 1]).tolist()
        gen_fps = gen_cm[0][1]
    else:
        gen_acc = gen_prec = gen_rec = gen_f1 = gen_nonhumor_prec = gen_nonhumor_rec = gen_nonhumor_f1 = None
        gen_cm = [[0, 0], [0, 0]]
        gen_fps = 0

    if len(y_cult) > 0:
        cult_acc = accuracy_score(y_true_cult, y_cult)
        cult_prec = precision_score(y_true_cult, y_cult, pos_label=1, zero_division=0)
        cult_rec = recall_score(y_true_cult, y_cult, pos_label=1, zero_division=0)
        cult_f1 = f1_score(y_true_cult, y_cult, pos_label=1, zero_division=0)
        cult_nonhumor_prec = precision_score(y_true_cult, y_cult, pos_label=0, zero_division=0)
        cult_nonhumor_rec = recall_score(y_true_cult, y_cult, pos_label=0, zero_division=0)
        cult_nonhumor_f1 = f1_score(y_true_cult, y_cult, pos_label=0, zero_division=0)
        cult_cm = confusion_matrix(y_true_cult, y_cult, labels=[0, 1]).tolist()
        cult_fps = cult_cm[0][1]
    else:
        cult_acc = cult_prec = cult_rec = cult_f1 = cult_nonhumor_prec = cult_nonhumor_rec = cult_nonhumor_f1 = None
        cult_cm = [[0, 0], [0, 0]]
        cult_fps = 0

    # Ground-truth probability breakdowns
    gt0_df = res_df[res_df["ground_truth"] == 0]
    gt1_df = res_df[res_df["ground_truth"] == 1]

    gen_prob_gt0 = round(float(gt0_df["general_humor_probability"].dropna().mean()), 4) if len(gt0_df) > 0 and gt0_df["general_humor_probability"].dropna().count() > 0 else None
    gen_prob_gt1 = round(float(gt1_df["general_humor_probability"].dropna().mean()), 4) if len(gt1_df) > 0 and gt1_df["general_humor_probability"].dropna().count() > 0 else None
    cult_prob_gt0 = round(float(gt0_df["cultural_humor_probability"].dropna().mean()), 4) if len(gt0_df) > 0 and gt0_df["cultural_humor_probability"].dropna().count() > 0 else None
    cult_prob_gt1 = round(float(gt1_df["cultural_humor_probability"].dropna().mean()), 4) if len(gt1_df) > 0 and gt1_df["cultural_humor_probability"].dropna().count() > 0 else None

    # ─── Prediction-Change Breakdown ─────────────────────────────────────────
    both_valid = gen_valid_mask & cult_valid_mask
    valid_res_df = res_df[both_valid]
    gen_wrong_cult_correct = int(((valid_res_df["general_prediction"] != valid_res_df["ground_truth"]) & (valid_res_df["cultural_prediction"] == valid_res_df["ground_truth"])).sum())
    gen_correct_cult_wrong = int(((valid_res_df["general_prediction"] == valid_res_df["ground_truth"]) & (valid_res_df["cultural_prediction"] != valid_res_df["ground_truth"])).sum())
    both_correct = int(((valid_res_df["general_prediction"] == valid_res_df["ground_truth"]) & (valid_res_df["cultural_prediction"] == valid_res_df["ground_truth"])).sum())
    both_wrong = int(((valid_res_df["general_prediction"] != valid_res_df["ground_truth"]) & (valid_res_df["cultural_prediction"] != valid_res_df["ground_truth"])).sum())
    total_changed = int((valid_res_df["general_prediction"] != valid_res_df["cultural_prediction"]).sum())
    
    # ─── JSON parsing and failure statistics ─────────────────────────────────
    gen_attempts = sum(1 for r in all_results if r.get("condition") == "general")
    cult_attempts = sum(1 for r in all_results if r.get("condition") == "cultural")
    gen_json_failures = int((res_df["general_json_error"].notna() | res_df["general_prediction"].isna()).sum())
    cult_json_failures = int((res_df["cultural_json_error"].notna() | (res_df["cultural_prediction"].isna() & ~res_df["cultural_stage2_fallback_to_stage1"])).sum())
    cult_fallbacks = int(res_df["cultural_stage2_fallback_to_stage1"].sum())
    
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
        "evaluated_samples": evaluated_samples_count,
        "manifest_total_samples": manifest_samples_count,
        "total_samples": evaluated_samples_count,
        "humorous_ground_truth": eval_pos_count,
        "non_humorous_ground_truth": eval_neg_count,
        "manifest_humorous_samples": sample_pos_count,
        "manifest_non_humorous_samples": sample_neg_count,
        "reproducibility": {
            "seed": seed,
            "prompt_version_hash": config_hashes["prompt_version_hash"],
            "model_config_hash": config_hashes["model_config_hash"],
            "cultural_config_hash": config_hashes["cultural_config_hash"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        },
        "inference_reliability": {
            "total_attempts": len(all_results),
            "general_attempts": gen_attempts,
            "cultural_attempts": cult_attempts,
            "general_evaluated": int(gen_valid_mask.sum()),
            "cultural_evaluated": int(cult_valid_mask.sum()),
            "general_failures": gen_json_failures,
            "cultural_failures": cult_json_failures,
            "general_timeouts": int(gen_timed_out.sum()),
            "cultural_timeouts": int(cult_timed_out.sum()),
            "general_valid_rate": round(int(gen_valid_mask.sum()) / gen_attempts, 4) if gen_attempts > 0 else 0.0,
            "cultural_valid_rate": round(int(cult_valid_mask.sum()) / cult_attempts, 4) if cult_attempts > 0 else 0.0,
            "cultural_stage2_fallbacks": cult_fallbacks,
        },
        "json_parsing": {
            "general_failures": gen_json_failures,
            "cultural_failures": cult_json_failures,
            "cultural_stage2_fallbacks": cult_fallbacks,
            "total_inferences": evaluated_samples_count * 2,
            "manifest_total_inferences": manifest_samples_count * 2,
        },
        "timing": {
            "general_mean_s": round(np.mean(gen_durations), 1) if gen_durations else None,
            "general_max_s": round(max(gen_durations), 1) if gen_durations else None,
            "cultural_mean_s": round(np.mean(cult_durations), 1) if cult_durations else None,
            "cultural_max_s": round(max(cult_durations), 1) if cult_durations else None,
            "total_inference_time_s": round(sum(gen_durations) + sum(cult_durations), 1),
        },
        "general_mode_metrics": {
            "evaluated_inferences": int(gen_valid_mask.sum()),
            "accuracy": round(gen_acc, 4) if gen_acc is not None else None,
            "humor_precision": round(gen_prec, 4) if gen_prec is not None else None,
            "humor_recall": round(gen_rec, 4) if gen_rec is not None else None,
            "humor_f1": round(gen_f1, 4) if gen_f1 is not None else None,
            "non_humor_precision": round(gen_nonhumor_prec, 4) if gen_nonhumor_prec is not None else None,
            "non_humor_recall": round(gen_nonhumor_rec, 4) if gen_nonhumor_rec is not None else None,
            "non_humor_f1": round(gen_nonhumor_f1, 4) if gen_nonhumor_f1 is not None else None,
            "non_humorous_to_humorous_errors": gen_fps,
            "mean_humor_prob_for_gt_non_humorous": gen_prob_gt0,
            "mean_humor_prob_for_gt_humorous": gen_prob_gt1,
            "confusion_matrix": gen_cm
        },
        "cultural_mode_metrics": {
            "evaluated_inferences": int(cult_valid_mask.sum()),
            "accuracy": round(cult_acc, 4) if cult_acc is not None else None,
            "humor_precision": round(cult_prec, 4) if cult_prec is not None else None,
            "humor_recall": round(cult_rec, 4) if cult_rec is not None else None,
            "humor_f1": round(cult_f1, 4) if cult_f1 is not None else None,
            "non_humor_precision": round(cult_nonhumor_prec, 4) if cult_nonhumor_prec is not None else None,
            "non_humor_recall": round(cult_nonhumor_rec, 4) if cult_nonhumor_rec is not None else None,
            "non_humor_f1": round(cult_nonhumor_f1, 4) if cult_nonhumor_f1 is not None else None,
            "non_humorous_to_humorous_errors": cult_fps,
            "mean_humor_prob_for_gt_non_humorous": cult_prob_gt0,
            "mean_humor_prob_for_gt_humorous": cult_prob_gt1,
            "confusion_matrix": cult_cm
        },
        "delta": {
            "accuracy_change": round(cult_acc - gen_acc, 4) if (cult_acc is not None and gen_acc is not None) else None,
            "humor_f1_change": round(cult_f1 - gen_f1, 4) if (cult_f1 is not None and gen_f1 is not None) else None,
            "non_humor_f1_change": round(cult_nonhumor_f1 - gen_nonhumor_f1, 4) if (cult_nonhumor_f1 is not None and gen_nonhumor_f1 is not None) else None,
            "fp_error_change": (cult_fps - gen_fps) if (cult_fps is not None and gen_fps is not None) else None,
        },
        "prediction_changes": {
            "general_wrong_to_cultural_correct": gen_wrong_cult_correct,
            "general_correct_to_cultural_wrong": gen_correct_cult_wrong,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "total_predictions_changed": total_changed,
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
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Stop after reaching this number of samples (0-indexed limit)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to sample manifest JSON file")
    parser.add_argument("--config-mode", type=str, default="both",
                        choices=["both", "general", "cultural"],
                        help="Evaluation configuration mode")
    parser.add_argument("--stage1-dir", type=str, default=None,
                        help="Directory containing completed Stage-1 results for Cultural mode")
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
            max_samples=args.max_samples,
            manifest_path=args.manifest,
            config_mode=args.config_mode,
            stage1_dir=args.stage1_dir,
        )
