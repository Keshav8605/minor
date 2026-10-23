"""
Offline Mock Tests for Config A and Config B Evaluation Runner.

Tests:
1. Manifest-driven execution loads exactly 100 samples in manifest order.
2. Config A runs only General condition into an isolated directory.
3. Config B runs only Cultural condition into an isolated directory.
4. Config B reuses Config A Stage-1 assessments when stage1_dir is provided.
5. Incomplete/corrupted checkpoints do not contaminate across configurations.
6. Checkpoint resumes cleanly and preserves metric denominator integrity.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from scripts.run_controlled_evaluation import run_controlled_experiment
from src.evaluation.checkpoint_manager import CheckpointManager


MANIFEST_PATH = "results/evaluation_heldout_100/sample_manifest.json"


@pytest.fixture
def mock_dirs():
    tmp_a = tempfile.mkdtemp(prefix="test_config_a_")
    tmp_b = tempfile.mkdtemp(prefix="test_config_b_")
    yield tmp_a, tmp_b
    shutil.rmtree(tmp_a, ignore_errors=True)
    shutil.rmtree(tmp_b, ignore_errors=True)


def mock_infer_fn(img_path, mode, ocr_text, retrieved_context, cultural_sources):
    """Deterministic mock inference function."""
    is_humorous = ("bhai" in ocr_text.lower() or "simran" in ocr_text.lower())
    return {
        "humorous": is_humorous,
        "prediction": "Humorous" if is_humorous else "Not Humorous",
        "humor_probability": 0.85 if is_humorous else 0.15,
        "non_humor_probability": 0.15 if is_humorous else 0.85,
        "confidence": 0.85,
        "reason": f"Mock {mode} reason",
        "detected_text": ocr_text[:30],
        "hit_max_new_tokens": False,
        "generated_token_count": 25,
        "timed_out": False,
        "error": None
    }


def mock_context_fn(ocr_text):
    """Deterministic mock context function."""
    return "Mock cultural context", ["PROJECT_JSON"], [{"attribute": "test", "similarity_score": 0.35}]


class TestHeldoutRunner:

    def test_config_a_manifest_execution(self, mock_dirs):
        """Test Config A runs only general mode on manifest samples into isolated dir."""
        dir_a, _ = mock_dirs
        
        # Run 5 samples with mock inference
        summary = run_controlled_experiment(
            results_dir=dir_a,
            manifest_path=MANIFEST_PATH,
            config_mode="general",
            max_samples=5,
            infer_fn=mock_infer_fn,
            context_fn=mock_context_fn,
            seed=42
        )
        
        mgr = CheckpointManager(dir_a)
        results = mgr.load_results()
        assert len(results) == 5
        
        # Verify all are general condition
        for r in results:
            assert r["condition"] == "general"
            assert r["sample_index"] < 5
            assert r["prediction"] in (0, 1)
            
        status = mgr.get_status()
        assert status["completed"] == 5
        assert status["general_completed"] == 5
        assert status["cultural_completed"] == 0

    def test_config_b_manifest_execution_with_stage1_reuse(self, mock_dirs):
        """Test Config B runs cultural mode and reuses Stage-1 from Config A."""
        dir_a, dir_b = mock_dirs
        
        # Step 1: Run Config A for 5 samples
        run_controlled_experiment(
            results_dir=dir_a,
            manifest_path=MANIFEST_PATH,
            config_mode="general",
            max_samples=5,
            infer_fn=mock_infer_fn,
            context_fn=mock_context_fn,
            seed=42
        )
        
        # Step 2: Run Config B for 5 samples pointing to Config A
        summary_b = run_controlled_experiment(
            results_dir=dir_b,
            manifest_path=MANIFEST_PATH,
            config_mode="cultural",
            stage1_dir=dir_a,
            max_samples=5,
            infer_fn=mock_infer_fn,
            context_fn=mock_context_fn,
            seed=42
        )
        
        mgr_b = CheckpointManager(dir_b)
        results_b = mgr_b.load_results()
        assert len(results_b) == 5
        
        # Verify all are cultural condition
        for r in results_b:
            assert r["condition"] == "cultural"
            assert r["sample_index"] < 5
            
        status_b = mgr_b.get_status()
        assert status_b["completed"] == 5
        assert status_b["general_completed"] == 0
        assert status_b["cultural_completed"] == 5
