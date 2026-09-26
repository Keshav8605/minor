"""
Tests for the Evaluation Checkpoint/Resume System.

Tests cover:
1. Save after one inference
2. Save after multiple inferences
3. Load checkpoint
4. Detect completed pairs
5. Skip completed pairs
6. Resume from correct pair
7. Interrupted inference is not marked complete
8. Duplicate prevention
9. Atomic checkpoint write
10. STOP behavior
11. CONTINUE behavior
12. STATUS behavior
13. Simulated process termination
14. Simulated restart and recovery

All tests use mock inference — no expensive Qwen VLM calls.
"""

import json
import os
import shutil
import tempfile
import time
import pytest

from src.evaluation.checkpoint_manager import CheckpointManager


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test results."""
    d = tempfile.mkdtemp(prefix="eval_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mgr(tmp_dir):
    """Create a CheckpointManager for testing."""
    return CheckpointManager(tmp_dir)


def _make_result(sample_index, condition, prediction=1, humor_prob=0.7):
    """Create a mock inference result record."""
    return {
        "sample_index": sample_index,
        "condition": condition,
        "meme_id": f"sample_{sample_index:03d}_test.jpg",
        "image_filename": "test.jpg",
        "image_hash": "abc123",
        "ocr_text": "test text",
        "ground_truth": 1,
        "prediction": prediction,
        "humor_probability": humor_prob,
        "non_humor_probability": 1 - humor_prob,
        "reasoning": "test reason",
        "json_error": None,
        "timed_out": False,
        "hit_max_tokens": False,
        "generated_token_count": 50,
        "inference_duration_s": 10.5,
        "cultural_category": None if condition == "general" else "family",
        "cultural_dependency": None if condition == "general" else "high",
        "cultural_sources": [],
        "retrieved_context": "",
        "drishtikon_matches_count": 0,
        "drishtikon_diagnostics": "[]",
        "context_passed_to_qwen": condition == "cultural",
        "prompt_hash": "abc123",
        "model_config_hash": "def456",
        "cultural_config_hash": "ghi789",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _make_checkpoint_args(completed_pairs, status="running"):
    """Create standard checkpoint save arguments."""
    return {
        "evaluation_id": "test_eval_001",
        "total_inferences": 10,
        "completed_pairs": completed_pairs,
        "sample_order": [
            {"sample_index": i, "image_filename": f"img_{i}.jpg", "ground_truth": i % 2}
            for i in range(5)
        ],
        "config_hashes": {
            "prompt_version_hash": "abc123",
            "model_config_hash": "def456",
            "cultural_config_hash": "ghi789",
        },
        "status": status,
    }


# ─── Test 1: Save after one inference ────────────────────────────────────────

class TestSingleInferencePersistence:
    def test_save_one_result(self, mgr):
        """After one inference, the result must be on disk."""
        result = _make_result(0, "general")
        mgr.append_result(result)
        
        loaded = mgr.load_results()
        assert len(loaded) == 1
        assert loaded[0]["sample_index"] == 0
        assert loaded[0]["condition"] == "general"
    
    def test_checkpoint_after_one(self, mgr):
        """Checkpoint must reflect one completed pair."""
        mgr.append_result(_make_result(0, "general"))
        mgr.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        
        ckpt = mgr.load_checkpoint()
        assert ckpt is not None
        assert ckpt["completed"] == 1
        assert ckpt["remaining"] == 9


# ─── Test 2: Save after multiple inferences ──────────────────────────────────

class TestMultipleInferencePersistence:
    def test_save_multiple_results(self, mgr):
        """Multiple results must all persist."""
        for i in range(3):
            mgr.append_result(_make_result(i, "general"))
            mgr.append_result(_make_result(i, "cultural"))
        
        loaded = mgr.load_results()
        assert len(loaded) == 6
    
    def test_checkpoint_multiple(self, mgr):
        """Checkpoint must reflect all completed pairs."""
        pairs = []
        for i in range(3):
            pairs.append((i, "general"))
            pairs.append((i, "cultural"))
        
        mgr.save_checkpoint(**_make_checkpoint_args(pairs))
        ckpt = mgr.load_checkpoint()
        assert ckpt["completed"] == 6
        assert ckpt["remaining"] == 4


# ─── Test 3: Load checkpoint ─────────────────────────────────────────────────

class TestLoadCheckpoint:
    def test_load_nonexistent(self, mgr):
        """Loading when no checkpoint exists returns None."""
        assert mgr.load_checkpoint() is None
    
    def test_load_existing(self, mgr):
        """Loading a saved checkpoint returns correct data."""
        mgr.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        ckpt = mgr.load_checkpoint()
        assert ckpt["evaluation_id"] == "test_eval_001"
        assert ckpt["total_inferences"] == 10
    
    def test_load_preserves_sample_order(self, mgr):
        """Sample order must survive save/load cycle."""
        args = _make_checkpoint_args([(0, "general")])
        mgr.save_checkpoint(**args)
        ckpt = mgr.load_checkpoint()
        assert len(ckpt["sample_order"]) == 5
        assert ckpt["sample_order"][0]["image_filename"] == "img_0.jpg"


# ─── Test 4: Detect completed pairs ─────────────────────────────────────────

class TestDetectCompletedPairs:
    def test_empty_initially(self, mgr):
        """No completed pairs when nothing persisted."""
        assert len(mgr.get_completed_pairs()) == 0
    
    def test_detect_from_jsonl(self, mgr):
        """Completed pairs are detected from JSONL results."""
        mgr.append_result(_make_result(0, "general"))
        mgr.append_result(_make_result(0, "cultural"))
        mgr.append_result(_make_result(1, "general"))
        
        completed = mgr.get_completed_pairs()
        assert (0, "general") in completed
        assert (0, "cultural") in completed
        assert (1, "general") in completed
        assert (1, "cultural") not in completed
        assert len(completed) == 3


# ─── Test 5: Skip completed pairs ───────────────────────────────────────────

class TestSkipCompleted:
    def test_is_pair_completed(self, mgr):
        """is_pair_completed correctly identifies persisted pairs."""
        mgr.append_result(_make_result(0, "general"))
        
        assert mgr.is_pair_completed(0, "general") is True
        assert mgr.is_pair_completed(0, "cultural") is False
        assert mgr.is_pair_completed(1, "general") is False


# ─── Test 6: Resume from correct pair ───────────────────────────────────────

class TestResumeFromCorrectPair:
    def test_resume_skips_completed(self, mgr):
        """Resume should identify the first unfinished pair."""
        # Simulate 3 completed inferences
        mgr.append_result(_make_result(0, "general"))
        mgr.append_result(_make_result(0, "cultural"))
        mgr.append_result(_make_result(1, "general"))
        
        completed = mgr.get_completed_pairs()
        
        # Build inference plan
        plan = []
        for i in range(5):
            plan.append((i, "general"))
            plan.append((i, "cultural"))
        
        # Find first unfinished
        next_pair = None
        for pair in plan:
            if pair not in completed:
                next_pair = pair
                break
        
        assert next_pair == (1, "cultural")


# ─── Test 7: Interrupted inference is not marked complete ────────────────────

class TestInterruptedInference:
    def test_incomplete_not_persisted(self, mgr):
        """An inference that wasn't persisted must not appear completed."""
        # Only persist sample 0 general
        mgr.append_result(_make_result(0, "general"))
        
        # Simulate: sample 0 cultural started but process crashed
        # (no append_result call)
        
        completed = mgr.get_completed_pairs()
        assert (0, "general") in completed
        assert (0, "cultural") not in completed
    
    def test_checkpoint_only_reflects_persisted(self, mgr):
        """Checkpoint must only contain pairs that were actually persisted."""
        mgr.append_result(_make_result(0, "general"))
        mgr.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        
        # Simulate crash: checkpoint says 1 completed
        ckpt = mgr.load_checkpoint()
        assert ckpt["completed"] == 1
        
        # JSONL confirms only 1 result
        assert len(mgr.load_results()) == 1


# ─── Test 8: Duplicate prevention ───────────────────────────────────────────

class TestDuplicatePrevention:
    def test_duplicate_detected(self, mgr):
        """Duplicate pairs should be detectable before inference."""
        mgr.append_result(_make_result(0, "general"))
        
        # Before running inference for (0, general) again:
        assert mgr.is_pair_completed(0, "general") is True
        # The evaluation loop should skip this pair


# ─── Test 9: Atomic checkpoint write ────────────────────────────────────────

class TestAtomicCheckpointWrite:
    def test_checkpoint_survives_multiple_saves(self, mgr):
        """Multiple checkpoint saves should not corrupt the file."""
        for i in range(10):
            pairs = [(j, "general") for j in range(i + 1)]
            mgr.save_checkpoint(**_make_checkpoint_args(pairs))
        
        ckpt = mgr.load_checkpoint()
        assert ckpt["completed"] == 10
    
    def test_no_temp_files_left(self, mgr):
        """After checkpoint save, no temp files should remain."""
        mgr.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        
        files = os.listdir(mgr.results_dir)
        temp_files = [f for f in files if f.endswith(".tmp")]
        assert len(temp_files) == 0


# ─── Test 10: STOP behavior ─────────────────────────────────────────────────

class TestStopBehavior:
    def test_stop_signal_creation(self, mgr):
        """request_stop creates a stop signal file."""
        assert mgr.is_stop_requested() is False
        mgr.request_stop()
        assert mgr.is_stop_requested() is True
    
    def test_stop_signal_clear(self, mgr):
        """clear_stop_signal removes the stop signal."""
        mgr.request_stop()
        assert mgr.is_stop_requested() is True
        mgr.clear_stop_signal()
        assert mgr.is_stop_requested() is False
    
    def test_stop_signal_file_exists(self, mgr):
        """Stop signal is a physical file on disk."""
        mgr.request_stop()
        assert os.path.exists(mgr.stop_signal_path)


# ─── Test 11: CONTINUE behavior ─────────────────────────────────────────────

class TestContinueBehavior:
    def test_continue_clears_stop(self, mgr):
        """On continue, stop signal should be cleared."""
        mgr.request_stop()
        mgr.clear_stop_signal()  # This is what CONTINUE does
        assert mgr.is_stop_requested() is False
    
    def test_continue_loads_correct_state(self, mgr):
        """After continue, checkpoint state reflects previous progress."""
        # Simulate: 3 pairs completed, then stopped
        for i in range(2):
            mgr.append_result(_make_result(i, "general"))
        mgr.append_result(_make_result(0, "cultural"))
        mgr.save_checkpoint(**_make_checkpoint_args(
            [(0, "general"), (0, "cultural"), (1, "general")],
            status="paused"
        ))
        
        # Simulate CONTINUE: load checkpoint
        ckpt = mgr.load_checkpoint()
        assert ckpt["status"] == "paused"
        assert ckpt["completed"] == 3
        
        completed = mgr.get_completed_pairs()
        assert len(completed) == 3


# ─── Test 12: STATUS behavior ───────────────────────────────────────────────

class TestStatusBehavior:
    def test_status_no_checkpoint(self, mgr):
        """Status with no checkpoint should report no_checkpoint."""
        status = mgr.get_status()
        assert status["status"] == "no_checkpoint"
        assert status["completed"] == 0
    
    def test_status_with_progress(self, mgr):
        """Status correctly reports progress."""
        mgr.append_result(_make_result(0, "general"))
        mgr.append_result(_make_result(0, "cultural"))
        mgr.save_checkpoint(**_make_checkpoint_args(
            [(0, "general"), (0, "cultural")],
            status="running"
        ))
        
        status = mgr.get_status()
        assert status["completed"] == 2
        assert status["general_completed"] == 1
        assert status["cultural_completed"] == 1
        assert status["status"] == "running"
    
    def test_status_does_not_modify(self, mgr):
        """Getting status must not modify any files."""
        mgr.append_result(_make_result(0, "general"))
        mgr.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        
        # Record file modification times
        ckpt_mtime = os.path.getmtime(mgr.checkpoint_path)
        results_mtime = os.path.getmtime(mgr.results_path)
        
        time.sleep(0.1)
        
        # Get status
        mgr.get_status()
        
        # Files should not be modified
        assert os.path.getmtime(mgr.checkpoint_path) == ckpt_mtime
        assert os.path.getmtime(mgr.results_path) == results_mtime


# ─── Test 13: Simulated process termination ──────────────────────────────────

class TestSimulatedTermination:
    def test_results_survive_new_manager(self, tmp_dir):
        """Results persisted by one manager instance are visible to another."""
        # Process 1: persist some results
        mgr1 = CheckpointManager(tmp_dir)
        mgr1.append_result(_make_result(0, "general"))
        mgr1.append_result(_make_result(0, "cultural"))
        mgr1.save_checkpoint(**_make_checkpoint_args(
            [(0, "general"), (0, "cultural")]
        ))
        
        # Simulate process crash: create new manager (like a new process)
        mgr2 = CheckpointManager(tmp_dir)
        
        # Process 2: should see the persisted results
        completed = mgr2.get_completed_pairs()
        assert (0, "general") in completed
        assert (0, "cultural") in completed
        assert len(completed) == 2
        
        ckpt = mgr2.load_checkpoint()
        assert ckpt["completed"] == 2
    
    def test_partial_crash_recovery(self, tmp_dir):
        """If inference starts but result isn't persisted, it's not completed."""
        mgr1 = CheckpointManager(tmp_dir)
        mgr1.append_result(_make_result(0, "general"))
        mgr1.save_checkpoint(**_make_checkpoint_args([(0, "general")]))
        
        # Simulate: sample 0 cultural inference started but crash before persist
        # (nothing appended for (0, "cultural"))
        
        # New process:
        mgr2 = CheckpointManager(tmp_dir)
        completed = mgr2.get_completed_pairs()
        
        assert (0, "general") in completed
        assert (0, "cultural") not in completed
        assert len(completed) == 1


# ─── Test 14: Simulated restart and recovery ─────────────────────────────────

class TestRestartAndRecovery:
    def test_full_resume_workflow(self, tmp_dir):
        """Complete workflow: run 5, stop, restart, resume, complete 10."""
        total_pairs = 10  # 5 samples × 2 conditions
        
        # Phase 1: Run 5 inferences
        mgr = CheckpointManager(tmp_dir)
        completed = []
        for i in range(3):
            for cond in ["general", "cultural"]:
                if len(completed) >= 5:
                    break
                mgr.append_result(_make_result(i, cond))
                completed.append((i, cond))
                mgr.save_checkpoint(**_make_checkpoint_args(completed))
            if len(completed) >= 5:
                break
        
        assert len(mgr.get_completed_pairs()) == 5
        
        # Phase 2: Simulate stop
        mgr.save_checkpoint(**_make_checkpoint_args(completed, status="paused"))
        
        # Phase 3: Simulate restart (new manager)
        mgr2 = CheckpointManager(tmp_dir)
        ckpt = mgr2.load_checkpoint()
        assert ckpt["status"] == "paused"
        assert ckpt["completed"] == 5
        
        existing_completed = mgr2.get_completed_pairs()
        assert len(existing_completed) == 5
        
        # Phase 4: Resume — run remaining 5
        plan = []
        for i in range(5):
            plan.append((i, "general"))
            plan.append((i, "cultural"))
        
        new_count = 0
        for pair in plan:
            if pair in existing_completed:
                continue
            mgr2.append_result(_make_result(pair[0], pair[1]))
            existing_completed.add(pair)
            new_count += 1
        
        # Should have executed exactly 5 new inferences
        assert new_count == 5
        
        # Total should be 10
        all_results = mgr2.load_results()
        assert len(all_results) == 10
        
        # No duplicates
        result_pairs = [(r["sample_index"], r["condition"]) for r in all_results]
        assert len(result_pairs) == len(set(result_pairs))
    
    def test_config_hash_consistency_check(self, tmp_dir):
        """Checkpoint stores config hashes for consistency verification."""
        mgr = CheckpointManager(tmp_dir)
        config = {
            "prompt_version_hash": "hash_v1",
            "model_config_hash": "model_v1",
            "cultural_config_hash": "cultural_v1",
        }
        mgr.save_checkpoint(
            evaluation_id="test",
            total_inferences=10,
            completed_pairs=[(0, "general")],
            sample_order=[],
            config_hashes=config,
        )
        
        ckpt = mgr.load_checkpoint()
        assert ckpt["config_hashes"]["prompt_version_hash"] == "hash_v1"


# ─── Test: JSONL resilience ──────────────────────────────────────────────────

class TestJSONLResilience:
    def test_corrupted_line_skipped(self, mgr):
        """A corrupted JSONL line should be skipped, not crash loading."""
        # Write valid result
        mgr.append_result(_make_result(0, "general"))
        
        # Manually append corrupted line
        with open(mgr.results_path, "a") as f:
            f.write("THIS IS NOT JSON\n")
        
        # Write another valid result
        mgr.append_result(_make_result(1, "general"))
        
        results = mgr.load_results()
        assert len(results) == 2  # Corrupted line skipped
    
    def test_empty_lines_skipped(self, mgr):
        """Empty lines in JSONL should be safely skipped."""
        mgr.append_result(_make_result(0, "general"))
        
        with open(mgr.results_path, "a") as f:
            f.write("\n\n")
        
        mgr.append_result(_make_result(1, "general"))
        
        results = mgr.load_results()
        assert len(results) == 2
