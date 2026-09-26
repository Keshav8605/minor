"""
Mock End-to-End Resume Simulation.

Simulates the full checkpoint/resume workflow using mock inference:
1. Run 5/10 mock pairs -> STOP
2. Verify checkpoint says 5/10
3. Terminate (new manager instance)
4. CONTINUE -> resume at 6/10
5. Verify first 5 are NOT re-executed
6. Complete remaining 5
7. Verify final = 10, 0 duplicates
"""

import os
import sys
import json
import shutil
import tempfile
import time
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.checkpoint_manager import CheckpointManager


def mock_infer(image_path, mode, ocr_text, retrieved_context, cultural_sources):
    """Mock inference function that returns deterministic results."""
    return {
        "humorous": True if "humorous" in image_path else False,
        "humor_probability": 0.75,
        "non_humor_probability": 0.25,
        "reason": f"Mock reason for {mode}",
        "cultural_category": "family" if mode == "cultural" else None,
        "cultural_dependency": "high" if mode == "cultural" else None,
        "error": None,
        "timed_out": False,
        "hit_max_new_tokens": False,
        "generated_token_count": 50,
    }


def mock_context(ocr_text):
    """Mock context builder."""
    return "Mock cultural context", ["PROJECT_JSON"], []


def run_simulation():
    print("=" * 60)
    print("MOCK END-TO-END RESUME SIMULATION")
    print("=" * 60)
    
    # Create temporary directory
    tmp_dir = tempfile.mkdtemp(prefix="mock_eval_")
    print(f"\nTemp directory: {tmp_dir}")
    
    try:
        # ─── Phase 1: Run 5/10 pairs then stop ──────────────────────────
        print("\n--- Phase 1: Run 5/10 pairs ---")
        
        mgr = CheckpointManager(tmp_dir)
        
        # Mock sample order (5 samples × 2 conditions = 10 pairs)
        sample_order = [
            {"sample_index": i, "image_filename": f"img_{i}.jpg", 
             "ground_truth": i % 2, "image_hash": f"hash_{i}"}
            for i in range(5)
        ]
        
        config_hashes = {
            "prompt_version_hash": "test_prompt_hash",
            "model_config_hash": "test_model_hash",
            "cultural_config_hash": "test_cultural_hash",
        }
        
        # Build inference plan
        plan = []
        for i in range(5):
            plan.append((i, "general"))
            plan.append((i, "cultural"))
        
        completed_pairs = set()
        executed_count = 0
        
        for sample_idx, condition in plan:
            if len(completed_pairs) >= 5:
                # Simulate STOP
                break
            
            if (sample_idx, condition) in completed_pairs:
                continue
            
            # Mock inference
            res = mock_infer(f"img_{sample_idx}.jpg", condition, "test", "", [])
            
            result_record = {
                "sample_index": sample_idx,
                "condition": condition,
                "meme_id": f"sample_{sample_idx:03d}_img_{sample_idx}.jpg",
                "image_filename": f"img_{sample_idx}.jpg",
                "image_hash": f"hash_{sample_idx}",
                "ocr_text": "test",
                "ground_truth": sample_idx % 2,
                "prediction": 1 if res["humorous"] else 0,
                "humor_probability": res["humor_probability"],
                "json_error": None,
                "timed_out": False,
                "inference_duration_s": 0.01,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            
            mgr.append_result(result_record)
            completed_pairs.add((sample_idx, condition))
            executed_count += 1
            
            mgr.save_checkpoint(
                evaluation_id="mock_eval",
                total_inferences=10,
                completed_pairs=sorted(completed_pairs, key=lambda x: (x[0], 0 if x[1] == "general" else 1)),
                sample_order=sample_order,
                config_hashes=config_hashes,
                status="running",
            )
            
            print(f"  Executed: ({sample_idx}, {condition}) -> {len(completed_pairs)}/10")
        
        # Save paused checkpoint
        mgr.save_checkpoint(
            evaluation_id="mock_eval",
            total_inferences=10,
            completed_pairs=sorted(completed_pairs, key=lambda x: (x[0], 0 if x[1] == "general" else 1)),
            sample_order=sample_order,
            config_hashes=config_hashes,
            status="paused",
        )
        
        print(f"\nPhase 1 result: {len(completed_pairs)}/10 completed")
        assert len(completed_pairs) == 5, f"Expected 5, got {len(completed_pairs)}"
        print("✓ Checkpoint says 5/10")
        
        # ─── Phase 2: Verify checkpoint ─────────────────────────────────
        print("\n--- Phase 2: Verify checkpoint on disk ---")
        
        ckpt = mgr.load_checkpoint()
        assert ckpt is not None
        assert ckpt["completed"] == 5
        assert ckpt["remaining"] == 5
        assert ckpt["status"] == "paused"
        print(f"  Status: {ckpt['status']}")
        print(f"  Completed: {ckpt['completed']}")
        print(f"  Remaining: {ckpt['remaining']}")
        print("✓ Checkpoint verified")
        
        # ─── Phase 3: Simulate process termination ──────────────────────
        print("\n--- Phase 3: Simulate process termination ---")
        del mgr  # Destroy manager
        print("  Manager destroyed (simulating process exit)")
        
        # ─── Phase 4: CONTINUE — new manager instance ───────────────────
        print("\n--- Phase 4: CONTINUE (new manager instance) ---")
        
        mgr2 = CheckpointManager(tmp_dir)
        mgr2.clear_stop_signal()
        
        ckpt2 = mgr2.load_checkpoint()
        assert ckpt2 is not None
        assert ckpt2["completed"] == 5
        
        existing_completed = mgr2.get_completed_pairs()
        general_done = sum(1 for _, c in existing_completed if c == "general")
        cultural_done = sum(1 for _, c in existing_completed if c == "cultural")
        
        print(f"  Loaded checkpoint: {len(existing_completed)}/10")
        print(f"  General: {general_done}/5")
        print(f"  Cultural: {cultural_done}/5")
        
        # Find next pair
        next_pair = None
        for pair in plan:
            if pair not in existing_completed:
                next_pair = pair
                break
        print(f"  Next pair: {next_pair}")
        assert next_pair is not None
        print("✓ Resume state verified")
        
        # ─── Phase 5: Execute remaining pairs ───────────────────────────
        print("\n--- Phase 5: Execute remaining 5 pairs ---")
        
        resume_executed = 0
        skipped = 0
        
        for sample_idx, condition in plan:
            if (sample_idx, condition) in existing_completed:
                skipped += 1
                continue
            
            res = mock_infer(f"img_{sample_idx}.jpg", condition, "test", "", [])
            
            result_record = {
                "sample_index": sample_idx,
                "condition": condition,
                "meme_id": f"sample_{sample_idx:03d}_img_{sample_idx}.jpg",
                "image_filename": f"img_{sample_idx}.jpg",
                "image_hash": f"hash_{sample_idx}",
                "ocr_text": "test",
                "ground_truth": sample_idx % 2,
                "prediction": 1 if res["humorous"] else 0,
                "humor_probability": res["humor_probability"],
                "json_error": None,
                "timed_out": False,
                "inference_duration_s": 0.01,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            
            mgr2.append_result(result_record)
            existing_completed.add((sample_idx, condition))
            resume_executed += 1
            
            print(f"  Executed: ({sample_idx}, {condition}) -> {len(existing_completed)}/10")
        
        print(f"\n  Skipped (already completed): {skipped}")
        print(f"  Newly executed: {resume_executed}")
        
        assert skipped == 5, f"Expected 5 skipped, got {skipped}"
        assert resume_executed == 5, f"Expected 5 new, got {resume_executed}"
        print("✓ First 5 pairs were NOT re-executed")
        
        # ─── Phase 6: Verify final state ────────────────────────────────
        print("\n--- Phase 6: Verify final state ---")
        
        all_results = mgr2.load_results()
        print(f"  Total results: {len(all_results)}")
        assert len(all_results) == 10, f"Expected 10, got {len(all_results)}"
        
        # Check for duplicates
        result_pairs = [(r["sample_index"], r["condition"]) for r in all_results]
        unique_pairs = set(result_pairs)
        duplicates = len(result_pairs) - len(unique_pairs)
        print(f"  Unique pairs: {len(unique_pairs)}")
        print(f"  Duplicates: {duplicates}")
        assert duplicates == 0, f"Found {duplicates} duplicates!"
        
        # Verify all 10 expected pairs are present
        expected_pairs = set(plan)
        assert unique_pairs == expected_pairs, f"Missing pairs: {expected_pairs - unique_pairs}"
        
        print("✓ 10/10 mock pairs completed, 0 duplicates")
        
        # ─── Final Summary ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Initial:                0/10")
        print(f"  After Phase 1 run:      5/10")
        print(f"  After STOP:             checkpoint saved (paused)")
        print(f"  After simulated restart: 5/10 (loaded from checkpoint)")
        print(f"  After CONTINUE:         resume at inference 6")
        print(f"  Final:                  10/10 mock pairs completed")
        print(f"  Duplicates:             0")
        print(f"  ALL ASSERTIONS PASSED:  ✓")
        print("=" * 60)
        
        return True
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)
