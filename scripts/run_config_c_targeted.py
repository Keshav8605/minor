"""
Targeted Config C Evaluator: Evaluates ONLY the 11 Disagreement Cases between Config A and Config B.

Targeted Sample Indices: [0, 4, 14, 24, 60, 65, 68, 72, 73, 77, 98]
- 5 cases where Config B degraded humour (GT=1, A=1, B=0)
- 6 cases where Config B corrected false positives (GT=0, A=1, B=0)

Reuses valid Stage-1 outputs from Config A.
Applies refined Config C cultural prompt instructions to test whether subtle satire/irony
can be preserved without losing corrected false positives.
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

from src.vlm.inference import VLMInferenceEngine
from src.cultural.context_builder import CulturalContextBuilder
from src.vlm.prompts import (
    build_humor_analysis_prompt,
    build_cultural_analysis_prompt_config_c,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("config_c_targeted")

DISAGREEMENT_INDICES = [0, 4, 14, 24, 60, 65, 68, 72, 73, 77, 98]


def _compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"


def _compute_prompt_version_hash() -> str:
    h = hashlib.sha256()
    general_msgs = build_humor_analysis_prompt("dummy.jpg")
    cultural_c_msgs = build_cultural_analysis_prompt_config_c("dummy.jpg", "dummy ocr", "dummy context")
    h.update(json.dumps(general_msgs, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(cultural_c_msgs, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage1_cache(stage1_dir: str):
    """Load and index completed Stage-1 results from Config A."""
    stage1_file = os.path.join(stage1_dir, "inference_results.jsonl")
    if not os.path.exists(stage1_file):
        raise FileNotFoundError(f"Stage 1 results file not found at: {stage1_file}")
    
    stage1_cache = {}
    with open(stage1_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                stage1_cache[rec["sample_index"]] = rec
    return stage1_cache


def run_targeted_evaluation(
    manifest_path: str = "results/evaluation_heldout_100/sample_manifest.json",
    image_dir: str = "data/processed/images",
    stage1_dir: str = "results/evaluation_heldout_100_config_a",
    results_dir: str = "results/evaluation_heldout_100_config_c_targeted",
    seed: int = 42,
    mock: bool = False,
    max_samples: int = None,
    sample_index_filter: list = None,
):
    os.makedirs(results_dir, exist_ok=True)
    results_jl = os.path.join(results_dir, "inference_results.jsonl")
    checkpoint_file = os.path.join(results_dir, "evaluation_checkpoint.json")

    # Load frozen manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_samples = manifest["samples"]

    # Load Stage-1 cache
    logger.info("Loading Stage-1 cache from %s...", stage1_dir)
    stage1_cache = load_stage1_cache(stage1_dir)

    # Compute configuration hashes
    prompt_hash = _compute_prompt_version_hash()
    model_config_hash = _compute_file_hash("configs/model.yaml")[:16]
    cultural_config_hash = _compute_file_hash("configs/cultural.yaml")[:16]

    config_hashes = {
        "prompt_version_hash": prompt_hash,
        "model_config_hash": model_config_hash,
        "cultural_config_hash": cultural_config_hash,
    }
    logger.info("Config C Prompt Version Hash: %s", prompt_hash)

    # Check already completed samples
    completed_indices = set()
    if os.path.exists(results_jl):
        with open(results_jl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed_indices.add(r["sample_index"])
    logger.info("Already completed in Config C: %d samples", len(completed_indices))

    target_indices = sample_index_filter if sample_index_filter is not None else DISAGREEMENT_INDICES
    if max_samples is not None:
        target_indices = target_indices[:max_samples]

    logger.info("Targeting %d disagreement samples: %s", len(target_indices), target_indices)

    # Initialize Engine & Context Builder if not mock
    engine = None
    builder = None
    if not mock:
        logger.info("Initializing VLM Engine for Config C...")
        engine = VLMInferenceEngine("configs/model.yaml")
        engine.load()

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
            min_similarity_threshold=cult_thresh,
        )

    for i, idx in enumerate(target_indices):
        if idx in completed_indices:
            logger.info("[%d/%d] Sample %d already completed. Skipping.", i + 1, len(target_indices), idx)
            continue

        sample_info = manifest_samples[idx]
        img_fn = sample_info["image_filename"]
        img_path = Path(image_dir) / img_fn
        if not img_path.exists():
            img_path = Path("Memotion 3/testImages/testImages") / img_fn
        gt = int(sample_info["ground_truth"])
        ocr_text = str(sample_info.get("ocr_text", ""))
        img_hash = sample_info.get("file_md5", "") or _compute_file_hash(str(img_path))[:16]

        # Validate Stage-1 cache entry
        s1_entry = stage1_cache.get(idx)
        if s1_entry is None or s1_entry.get("prediction") is None:
            raise RuntimeError(f"Missing valid Stage-1 result for target sample {idx} ({img_fn})")

        # Confirm image hash consistency
        s1_img_hash = s1_entry.get("image_hash", "")[:16]
        logger.info(
            "[%d/%d] Processing Sample %d (%s) | GT=%d | Stage-1 P(humor)=%s",
            i + 1, len(target_indices), idx, img_fn, gt, s1_entry.get("humor_probability")
        )

        stage1_result = {
            "humorous": bool(s1_entry["prediction"] == 1),
            "prediction": "Humorous" if s1_entry["prediction"] == 1 else "Not Humorous",
            "humor_probability": s1_entry.get("humor_probability"),
            "non_humor_probability": s1_entry.get("non_humor_probability"),
            "confidence": max(s1_entry.get("humor_probability") or 0.5, s1_entry.get("non_humor_probability") or 0.5),
            "reason": s1_entry.get("reasoning", ""),
            "detected_text": s1_entry.get("ocr_text", ""),
        }

        # Build cultural context
        if mock:
            cult_context = f"EXTERNAL CULTURAL CONTEXT:\nPROJECT CULTURAL KNOWLEDGE: Mock context for {img_fn}"
            cult_sources = ["PROJECT_JSON"]
            drish_matches = []
        else:
            cult_context, cult_sources, drish_matches = builder.build_context(ocr_text)

        _set_seed(seed)
        start_t = time.time()

        if mock:
            # Mock inference simulating recovering humour for satirical cases
            dur = 0.05
            is_timed_out = False
            has_error = False
            # Mock realistic outcome: recover satire cases (24, 65, 68, 73, 98), keep non-humour (0, 4, 14, 60, 72, 77)
            if gt == 1:
                pred_val = 1
                h_prob = 0.58
                nh_prob = 0.42
                reason = "Config C mock: Identified intentional comedic mechanism despite serious cultural context."
            else:
                pred_val = 0
                h_prob = 0.41
                nh_prob = 0.59
                reason = "Config C mock: No comedic mechanism found; confirmed non-humorous."
            res = {
                "humorous": bool(pred_val == 1),
                "prediction": "Humorous" if pred_val == 1 else "Not Humorous",
                "humor_probability": h_prob,
                "non_humor_probability": nh_prob,
                "reason": reason,
                "cultural_category": "Bollywood" if "Bollywood" in cult_context else "general",
                "cultural_dependency": "medium",
                "timed_out": False,
                "hit_max_new_tokens": False,
                "generated_token_count": 45,
            }
        else:
            res = engine.infer(
                str(img_path),
                mode="cultural_c",
                ocr_text=ocr_text,
                retrieved_context=cult_context,
                cultural_sources=cult_sources,
                stage1_result=stage1_result,
                two_stage=True,
            )
            dur = time.time() - start_t
            is_timed_out = bool(res.get("timed_out", False))
            has_error = bool(res.get("error") or res.get("stage2_error"))
            fallback_applied = bool(res.get("stage2_fallback_to_stage1", False))

            if is_timed_out or (has_error and not fallback_applied):
                pred_val = None
            elif res.get("humorous") is True:
                pred_val = 1
            elif res.get("humorous") is False:
                pred_val = 0
            else:
                pred_val = None

        record = {
            "sample_index": idx,
            "condition": "cultural_c",
            "meme_id": f"sample_{idx:03d}_{img_fn}",
            "image_filename": img_fn,
            "image_hash": img_hash[:16],
            "ocr_text": ocr_text,
            "ground_truth": gt,
            "prediction": pred_val,
            "humor_probability": res.get("humor_probability") if pred_val is not None else None,
            "non_humor_probability": res.get("non_humor_probability") if pred_val is not None else None,
            "reasoning": res.get("reason", "") or ("Inference timed out" if is_timed_out else "Inference failed"),
            "json_error": res.get("error") or ("TIMEOUT" if is_timed_out else None),
            "timed_out": is_timed_out,
            "hit_max_tokens": res.get("hit_max_new_tokens", False),
            "generated_token_count": res.get("generated_token_count"),
            "inference_duration_s": round(dur, 2),
            "cultural_category": res.get("cultural_category"),
            "cultural_dependency": res.get("cultural_dependency"),
            "cultural_sources": cult_sources,
            "retrieved_context": cult_context,
            "context_passed_to_qwen": res.get("context_passed_to_qwen", True),
            "two_stage_applied": True,
            "prompt_hash": prompt_hash,
            "model_config_hash": model_config_hash,
            "cultural_config_hash": cultural_config_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        # Atomic append to JSONL
        with open(results_jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        completed_indices.add(idx)
        logger.info(
            "Completed sample %d: Pred=%s (P=%s) in %.1fs",
            idx, pred_val, res.get("humor_probability"), dur
        )

    # Save final checkpoint
    ckpt = {
        "total_target_samples": len(target_indices),
        "completed": len(completed_indices),
        "target_indices": target_indices,
        "completed_indices": sorted(list(completed_indices)),
        "config_hashes": config_hashes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    ckpt_tmp = checkpoint_file + ".tmp"
    with open(ckpt_tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(ckpt_tmp, checkpoint_file)
    logger.info("Targeted evaluation complete. Checkpoint saved to %s", checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted Config C Evaluator")
    parser.add_argument("--mock", action="store_true", help="Run with mock inference for fast pipeline verification")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--results-dir", type=str, default="results/evaluation_heldout_100_config_c_targeted")
    args = parser.parse_args()

    run_targeted_evaluation(
        mock=args.mock,
        max_samples=args.max_samples,
        results_dir=args.results_dir,
    )
