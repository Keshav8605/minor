"""
Evaluation Checkpoint Manager.

Provides crash-safe persistent checkpoint and incremental result storage
for the controlled evaluation. Survives laptop shutdown, server restart,
and unexpected process termination.

Architecture:
- JSONL file for incremental inference results (append-only)
- JSON checkpoint file for evaluation state (atomic write-replace)
- Completed pairs tracked as (sample_index, condition) tuples

RESEARCH INTEGRITY:
- This module is ONLY execution infrastructure
- It does NOT modify inference logic, prompts, metrics, or methodology
"""

import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("checkpoint_manager")


class CheckpointManager:
    """
    Manages persistent checkpoint state and incremental result storage.
    
    Files managed:
    - evaluation_checkpoint.json: Current evaluation state (atomic writes)
    - inference_results.jsonl: Append-only log of completed inference results
    - stop_signal.flag: Presence indicates a stop request
    """
    
    CHECKPOINT_FILE = "evaluation_checkpoint.json"
    RESULTS_FILE = "inference_results.jsonl"
    STOP_SIGNAL_FILE = "stop_signal.flag"
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(results_dir, self.CHECKPOINT_FILE)
        self.results_path = os.path.join(results_dir, self.RESULTS_FILE)
        self.stop_signal_path = os.path.join(results_dir, self.STOP_SIGNAL_FILE)
    
    # ─── Atomic Write ───────────────────────────────────────────────────────
    
    def _atomic_write_json(self, filepath: str, data: dict):
        """
        Write JSON atomically: write to temp file, flush, fsync, then rename.
        If the process is killed during write, the original file remains intact.
        """
        dirpath = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp", prefix=".ckpt_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace (on Windows, need to remove target first)
            if os.path.exists(filepath):
                os.replace(tmp_path, filepath)
            else:
                os.rename(tmp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    
    # ─── JSONL Append ────────────────────────────────────────────────────────
    
    def append_result(self, result: dict):
        """
        Append a single inference result to the JSONL file.
        Flush and fsync immediately to ensure persistence.
        """
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
    
    def load_results(self) -> List[dict]:
        """Load all persisted inference results from JSONL."""
        results = []
        if not os.path.exists(self.results_path):
            return results
        with open(self.results_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupted JSONL line %d", line_num)
        return results
    
    # ─── Checkpoint State ────────────────────────────────────────────────────
    
    def save_checkpoint(
        self,
        evaluation_id: str,
        total_inferences: int,
        completed_pairs: List[Tuple[int, str]],
        sample_order: List[dict],
        config_hashes: dict,
        status: str = "running",
        timeouts: int = 0,
        json_failures: int = 0,
        start_time: Optional[float] = None,
    ):
        """Save evaluation checkpoint atomically."""
        checkpoint = {
            "evaluation_id": evaluation_id,
            "total_inferences": total_inferences,
            "completed": len(completed_pairs),
            "remaining": total_inferences - len(completed_pairs),
            "completed_pairs": [
                {"sample_index": idx, "condition": cond}
                for idx, cond in completed_pairs
            ],
            "last_completed_sample": completed_pairs[-1][0] if completed_pairs else None,
            "last_completed_condition": completed_pairs[-1][1] if completed_pairs else None,
            "status": status,
            "timeouts": timeouts,
            "json_failures": json_failures,
            "config_hashes": config_hashes,
            "sample_order": sample_order,
            "start_time": start_time,
            "last_updated": time.time(),
            "last_updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._atomic_write_json(self.checkpoint_path, checkpoint)
        logger.info(
            "Checkpoint saved: %d/%d completed (%s)",
            len(completed_pairs), total_inferences, status
        )
    
    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint from disk. Returns None if no checkpoint exists."""
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load checkpoint: %s", e)
            return None
    
    @staticmethod
    def is_record_complete(r: dict) -> bool:
        """
        Check if an inference result record is structurally complete.
        An incomplete record (missing critical identifiers, or marked incomplete)
        must not be treated as completed.
        """
        if not isinstance(r, dict):
            return False
        if r.get("incomplete", False):
            return False
        required_keys = {"sample_index", "condition", "image_filename"}
        if not required_keys.issubset(r.keys()):
            return False
        if r.get("sample_index") is None or not r.get("condition"):
            return False
        return True

    @staticmethod
    def is_record_valid_prediction(r: dict) -> bool:
        """
        Check if an inference result record represents a valid, successful prediction
        (not timed out, no json/parsing error, and prediction in {0, 1}).
        """
        if not CheckpointManager.is_record_complete(r):
            return False
        if r.get("timed_out", False):
            return False
        if r.get("json_error") is not None:
            return False
        return r.get("prediction") in (0, 1)

    def validate_configuration(self, expected_hashes: dict, config_name: Optional[str] = None):
        """
        Ensure results from different configurations cannot share checkpoints accidentally.
        Raises ValueError if existing checkpoint config hashes mismatch expected hashes.
        """
        ckpt = self.load_checkpoint()
        if not ckpt:
            return
        
        ckpt_hashes = ckpt.get("config_hashes", {})
        mismatches = []
        for k, v in expected_hashes.items():
            if k in ckpt_hashes and ckpt_hashes[k] != v:
                mismatches.append(f"{k}: existing '{ckpt_hashes[k]}' != expected '{v}'")
        
        if mismatches:
            raise ValueError(
                f"Configuration mismatch in checkpoint directory '{self.results_dir}':\n"
                f"  {', '.join(mismatches)}\n"
                "Results from different configurations cannot share checkpoints accidentally. "
                "Specify a separate results_dir for each configuration (e.g. config_a, config_b, config_c)."
            )

    def get_completed_pairs(self) -> Set[Tuple[int, str]]:
        """
        Get the set of completed (sample_index, condition) pairs.
        Cross-references checkpoint with actual JSONL results for safety.
        Only structurally complete records in JSONL are considered completed.
        Incomplete or corrupt records are rejected so they are not skipped.
        """
        completed = set()
        
        # Primary source: JSONL results (ground truth of what was persisted)
        results = self.load_results()
        for r in results:
            if self.is_record_complete(r):
                sample_idx = r.get("sample_index")
                condition = r.get("condition")
                completed.add((sample_idx, condition))
        
        return completed

    def get_valid_completed_pairs(self) -> Set[Tuple[int, str]]:
        """
        Get the set of (sample_index, condition) pairs that produced valid predictions.
        Failed/timed-out inferences are excluded so resume logic never mistakes
        a failure for a completed valid prediction.
        """
        valid_completed = set()
        results = self.load_results()
        for r in results:
            if self.is_record_valid_prediction(r):
                valid_completed.add((r["sample_index"], r["condition"]))
        return valid_completed
    
    def is_pair_completed(self, sample_index: int, condition: str) -> bool:
        """Check if a specific (sample_index, condition) pair is completed."""
        completed = self.get_completed_pairs()
        return (sample_index, condition) in completed
    
    # ─── Stop Signal ─────────────────────────────────────────────────────────
    
    def request_stop(self):
        """Create a stop signal file."""
        with open(self.stop_signal_path, "w") as f:
            f.write(f"stop_requested_at={time.time()}\n")
            f.flush()
            os.fsync(f.fileno())
        logger.info("Stop signal created at %s", self.stop_signal_path)
    
    def is_stop_requested(self) -> bool:
        """Check if a stop signal exists."""
        return os.path.exists(self.stop_signal_path)
    
    def clear_stop_signal(self):
        """Remove the stop signal file."""
        if os.path.exists(self.stop_signal_path):
            os.remove(self.stop_signal_path)
            logger.info("Stop signal cleared")
    
    # ─── Status Report ───────────────────────────────────────────────────────
    
    def get_status(self) -> dict:
        """Generate a status report from persisted state."""
        checkpoint = self.load_checkpoint()
        completed_pairs = self.get_completed_pairs()
        valid_pairs = self.get_valid_completed_pairs()
        
        general_completed = sum(1 for _, c in completed_pairs if c == "general")
        cultural_completed = sum(1 for _, c in completed_pairs if c == "cultural")
        general_valid = sum(1 for _, c in valid_pairs if c == "general")
        cultural_valid = sum(1 for _, c in valid_pairs if c == "cultural")
        
        # Scan actual results for true failure counters
        results = self.load_results()
        timeouts = sum(1 for r in results if self.is_record_complete(r) and r.get("timed_out", False))
        json_failures = sum(1 for r in results if self.is_record_complete(r) and r.get("json_error") is not None)
        
        total = checkpoint.get("total_inferences", 40) if checkpoint else 40
        
        return {
            "completed": len(completed_pairs),
            "valid_completed": len(valid_pairs),
            "total": total,
            "remaining": total - len(completed_pairs),
            "general_completed": general_completed,
            "cultural_completed": cultural_completed,
            "general_valid": general_valid,
            "cultural_valid": cultural_valid,
            "general_total": total // 2,
            "cultural_total": total // 2,
            "last_completed_sample": checkpoint.get("last_completed_sample") if checkpoint else None,
            "last_completed_condition": checkpoint.get("last_completed_condition") if checkpoint else None,
            "status": checkpoint.get("status", "unknown") if checkpoint else "no_checkpoint",
            "timeouts": timeouts if timeouts > 0 else (checkpoint.get("timeouts", 0) if checkpoint else 0),
            "json_failures": json_failures if json_failures > 0 else (checkpoint.get("json_failures", 0) if checkpoint else 0),
            "evaluation_dir": self.results_dir,
            "checkpoint_exists": checkpoint is not None,
            "stop_requested": self.is_stop_requested(),
            "start_time": checkpoint.get("start_time") if checkpoint else None,
        }
