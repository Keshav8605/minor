"""
Offline Unit and Integration Tests for Critical Evaluation Safety Fixes.

Covers:
1. Timeout is recorded as failure, never class 0.
2. JSON parsing failures remain invalid rather than negative predictions.
3. Failed predictions are excluded from accuracy and F1 denominators.
4. Valid completed results survive interruption and resume.
5. Incomplete records are not treated as completed.
6. Results from different configurations cannot share checkpoints accidentally.
7. Pure PIL/NumPy perceptual-hash and OCR deduplication integrity.
"""

import os
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.checkpoint_manager import CheckpointManager
from src.evaluation.prediction_store import PredictionStore
from src.evaluation.deduplication import (
    compute_average_hash,
    hamming_distance,
    compute_ocr_similarity,
    normalize_ocr_text,
    check_candidate_against_quarantine,
)
from src.vlm.output_parser import parse_json_response
from src.vlm.inference import VLMInferenceEngine
from scripts.run_controlled_evaluation import run_controlled_experiment, _generate_final_outputs


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_eval_dir():
    """Create a temporary directory for evaluation outputs."""
    d = tempfile.mkdtemp(prefix="test_safety_eval_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── Test 1: Timeout is recorded as failure, never class 0 ──────────────────

class TestTimeoutFailureHandling:
    def test_timeout_in_vlm_engine_sets_unavailable_not_class_0(self):
        """Simulate VLM timeout: verify prediction is 'Unavailable' and humorous is None."""
        engine = VLMInferenceEngine("configs/model.yaml")
        diagnostics = {
            "timed_out": True,
            "hit_max_new_tokens": False,
            "generated_token_count": 50,
            "raw_response": '{"humorous": false',  # Truncated due to timeout
            "inference_duration_s": 1500.0,
            "prefill_duration_s": 300.0,
            "decoding_duration_s": 1200.0,
        }
        parsed = {"humorous": False, "reason": "Interrupted", "prediction": "Not Humorous"}
        # Even if parsed had humorous=False before timeout occurred:
        res = engine._build_result(parsed, humor_prob=0.1, non_humor_prob=0.9, mode="general", diagnostics=diagnostics)
        
        assert res["timed_out"] is True
        assert res["humorous"] is None, "Timed out inference MUST have humorous=None"
        assert res["prediction"] == "Unavailable", "Timed out inference MUST have prediction='Unavailable'"
        assert res["humor_probability"] is None, "Timed out inference MUST NOT keep probabilities"

    def test_timeout_in_controlled_eval_records_prediction_none(self, tmp_eval_dir):
        """In run_controlled_evaluation, a timed out inference must record prediction=None, never 0."""
        # Mock infer_fn returning timeout
        def mock_infer_timeout(img_path, mode, ocr, cult_ctx, cult_sources):
            return {
                "humorous": None,
                "prediction": "Unavailable",
                "error": "GENERATION_TIMEOUT",
                "timed_out": True,
                "reason": "Process timed out",
                "inference_duration_s": 1500.0,
            }
            
        run_controlled_experiment(
            test_csv_path="data/processed/test.csv",
            sample_size=2,
            results_dir=tmp_eval_dir,
            infer_fn=mock_infer_timeout,
            max_samples=1,
        )
        
        mgr = CheckpointManager(tmp_eval_dir)
        results = mgr.load_results()
        assert len(results) == 2  # general and cultural
        for r in results:
            assert r["timed_out"] is True
            assert r["prediction"] is None, f"Failed inference must record prediction=None, got {r['prediction']}"
            assert r["json_error"] is not None


# ─── Test 2: JSON parsing failures remain invalid rather than negative ──────

class TestJSONParsingFailures:
    def test_unparseable_output_returns_error_result_with_none_prediction(self):
        """Verify output_parser produces humorous=None for corrupt/non-JSON text."""
        corrupt_text = "I am a language model and I refuse to output JSON. The meme is not funny."
        parsed = parse_json_response(corrupt_text)
        
        assert parsed["error"] is not None
        assert parsed["humorous"] is None
        assert parsed["prediction"] == "Unavailable"
        
    def test_json_failure_in_controlled_eval_persists_none_prediction(self, tmp_eval_dir):
        """Verify run_controlled_evaluation stores prediction=None on parsing failure."""
        def mock_infer_json_fail(img_path, mode, ocr, cult_ctx, cult_sources):
            return {
                "humorous": None,
                "prediction": "Unavailable",
                "error": "Failed to parse JSON",
                "timed_out": False,
                "reason": "Invalid JSON syntax",
            }
            
        run_controlled_experiment(
            test_csv_path="data/processed/test.csv",
            sample_size=2,
            results_dir=tmp_eval_dir,
            infer_fn=mock_infer_json_fail,
            max_samples=1,
        )
        
        mgr = CheckpointManager(tmp_eval_dir)
        results = mgr.load_results()
        assert len(results) == 2
        for r in results:
            assert r["prediction"] is None
            assert r["json_error"] == "Failed to parse JSON"


# ─── Test 3: Failed predictions excluded from accuracy and F1 denominators ──

class TestMetricDenominatorsExcludeFailures:
    def test_metrics_exclude_timeouts_and_errors(self, tmp_eval_dir):
        """Construct a mixed dataset of 5 valid, 3 timed-out, 2 error results; assert denominator=5."""
        mgr = CheckpointManager(tmp_eval_dir)
        eval_df = pd.DataFrame([
            {"image_filename": f"img_{i}.jpg", "is_humorous": i % 2, "ocr": f"text {i}"}
            for i in range(10)
        ])
        
        records = []
        # 5 valid samples (indices 0..4): 3 correct, 2 incorrect
        # index 0: gt=0, pred=0 (correct)
        # index 1: gt=1, pred=1 (correct)
        # index 2: gt=0, pred=1 (incorrect)
        # index 3: gt=1, pred=1 (correct)
        # index 4: gt=0, pred=1 (incorrect)
        valid_preds = [0, 1, 1, 1, 1]
        for i in range(5):
            for cond in ["general", "cultural"]:
                r = {
                    "sample_index": i,
                    "condition": cond,
                    "meme_id": f"sample_{i:03d}_img_{i}.jpg",
                    "image_filename": f"img_{i}.jpg",
                    "image_hash": "hash123",
                    "ocr_text": f"text {i}",
                    "ground_truth": i % 2,
                    "prediction": valid_preds[i],
                    "humor_probability": 0.8 if valid_preds[i] == 1 else 0.2,
                    "non_humor_probability": 0.2 if valid_preds[i] == 1 else 0.8,
                    "reasoning": "Valid reasoning",
                    "json_error": None,
                    "timed_out": False,
                    "inference_duration_s": 500.0,
                }
                records.append(r)
                mgr.append_result(r)
                
        # 3 timed-out samples (indices 5, 6, 7)
        for i in range(5, 8):
            for cond in ["general", "cultural"]:
                r = {
                    "sample_index": i,
                    "condition": cond,
                    "meme_id": f"sample_{i:03d}_img_{i}.jpg",
                    "image_filename": f"img_{i}.jpg",
                    "image_hash": "hash123",
                    "ocr_text": f"text {i}",
                    "ground_truth": i % 2,
                    "prediction": None,  # Properly None
                    "humor_probability": None,
                    "non_humor_probability": None,
                    "reasoning": "Inference timed out",
                    "json_error": "TIMEOUT",
                    "timed_out": True,
                    "inference_duration_s": 1500.0,
                }
                records.append(r)
                mgr.append_result(r)
                
        # 2 JSON parsing errors (indices 8, 9)
        for i in range(8, 10):
            for cond in ["general", "cultural"]:
                r = {
                    "sample_index": i,
                    "condition": cond,
                    "meme_id": f"sample_{i:03d}_img_{i}.jpg",
                    "image_filename": f"img_{i}.jpg",
                    "image_hash": "hash123",
                    "ocr_text": f"text {i}",
                    "ground_truth": i % 2,
                    "prediction": None,
                    "humor_probability": None,
                    "non_humor_probability": None,
                    "reasoning": "Parsing error",
                    "json_error": "Failed to parse JSON",
                    "timed_out": False,
                    "inference_duration_s": 50.0,
                }
                records.append(r)
                mgr.append_result(r)
                
        config_hashes = {
            "prompt_version_hash": "p_hash",
            "model_config_hash": "m_hash",
            "cultural_config_hash": "c_hash",
        }
        sample_order = [{"sample_index": i, "image_filename": f"img_{i}.jpg"} for i in range(10)]
        
        summary = _generate_final_outputs(
            results_dir=tmp_eval_dir,
            mgr=mgr,
            eval_df=eval_df,
            sample_order=sample_order,
            config_hashes=config_hashes,
            sample_pos_count=5,
            sample_neg_count=5,
            seed=42,
        )
        
        # Assertions
        # 1. Total samples evaluated = 10, but valid inferences = 5
        assert summary["general_mode_metrics"]["evaluated_inferences"] == 5
        assert summary["cultural_mode_metrics"]["evaluated_inferences"] == 5
        
        # 2. Accuracy: 3 correct out of 5 valid = 3/5 = 0.60
        assert summary["general_mode_metrics"]["accuracy"] == 0.60
        assert summary["cultural_mode_metrics"]["accuracy"] == 0.60
        
        # 3. Reliability section must explicitly capture timeouts and failures
        rel = summary["inference_reliability"]
        assert rel["general_attempts"] == 10
        assert rel["general_evaluated"] == 5
        assert rel["general_failures"] == 5
        assert rel["general_timeouts"] == 3
        assert rel["general_valid_rate"] == 0.50


# ─── Test 4: Valid completed results survive interruption and resume ────────

class TestInterruptionAndResume:
    def test_resume_preserves_valid_and_avoids_duplicate_inference(self, tmp_eval_dir):
        """Ensure restarting an evaluation continues exactly from the next unfinished sample."""
        mgr = CheckpointManager(tmp_eval_dir)
        eval_df = pd.DataFrame([
            {"image_filename": f"img_{i}.jpg", "is_humorous": i % 2, "ocr": f"text {i}"}
            for i in range(4)
        ])
        
        call_counts = {}
        def mock_infer_tracking(img_path, mode, ocr, cult_ctx, cult_sources):
            call_counts[(img_path, mode)] = call_counts.get((img_path, mode), 0) + 1
            return {
                "humorous": True,
                "prediction": "Humorous",
                "humor_probability": 0.8,
                "non_humor_probability": 0.2,
                "reason": "OK",
            }
            
        # Step 1: Run 1 sample (2 inferences)
        run_controlled_experiment(
            test_csv_path="data/processed/test.csv",
            sample_size=2,
            results_dir=tmp_eval_dir,
            infer_fn=mock_infer_tracking,
            max_samples=1,
        )
        
        assert len(mgr.get_completed_pairs()) == 2  # (0, gen), (0, cult)
        assert len(call_counts) == 2
        
        # Step 2: Resume with max_samples=2 (runs remaining sample 1)
        run_controlled_experiment(
            test_csv_path="data/processed/test.csv",
            sample_size=2,
            results_dir=tmp_eval_dir,
            infer_fn=mock_infer_tracking,
            resume=True,
            max_samples=2,
        )
        
        # Check that none of the first sample's inferences were called a second time
        for k, v in call_counts.items():
            assert v == 1, f"Inference for {k} was executed {v} times instead of exactly once!"
            
        assert len(mgr.get_completed_pairs()) == 4


# ─── Test 5: Incomplete records are not treated as completed ────────────────

class TestIncompleteRecordRejection:
    def test_corrupt_or_incomplete_records_ignored_by_checkpoint_manager(self, tmp_eval_dir):
        """Incomplete records (e.g. truncated JSON, missing condition, marked incomplete) must be rejected."""
        mgr = CheckpointManager(tmp_eval_dir)
        
        # 1. Valid record
        valid_rec = {
            "sample_index": 0,
            "condition": "general",
            "image_filename": "img_0.jpg",
            "prediction": 1,
            "ground_truth": 1,
        }
        mgr.append_result(valid_rec)
        
        # 2. Incomplete record: missing condition
        incomp_rec_1 = {
            "sample_index": 1,
            "image_filename": "img_1.jpg",
        }
        mgr.append_result(incomp_rec_1)
        
        # 3. Incomplete record: missing sample_index
        incomp_rec_2 = {
            "condition": "cultural",
            "image_filename": "img_2.jpg",
        }
        mgr.append_result(incomp_rec_2)
        
        # 4. Incomplete record: explicit incomplete flag
        incomp_rec_3 = {
            "sample_index": 3,
            "condition": "general",
            "image_filename": "img_3.jpg",
            "incomplete": True,
        }
        mgr.append_result(incomp_rec_3)
        
        completed = mgr.get_completed_pairs()
        assert (0, "general") in completed
        assert (1, "general") not in completed
        assert (2, "cultural") not in completed
        assert (3, "general") not in completed
        assert len(completed) == 1


# ─── Test 6: Results from different configurations cannot share checkpoints ─

class TestConfigurationIsolation:
    def test_configuration_mismatch_raises_error(self, tmp_eval_dir):
        """Resuming with different config hashes must raise ValueError to prevent checkpoint collision."""
        mgr = CheckpointManager(tmp_eval_dir)
        config_a = {
            "prompt_version_hash": "prompt_hash_A",
            "model_config_hash": "model_hash_v1",
            "cultural_config_hash": "cult_hash_v1",
        }
        mgr.save_checkpoint(
            evaluation_id="test_run",
            total_inferences=10,
            completed_pairs=[(0, "general")],
            sample_order=[],
            config_hashes=config_a,
        )
        
        # Attempt to validate under Config B (different prompt hash)
        config_b = {
            "prompt_version_hash": "prompt_hash_B",
            "model_config_hash": "model_hash_v1",
            "cultural_config_hash": "cult_hash_v1",
        }
        
        with pytest.raises(ValueError) as excinfo:
            mgr.validate_configuration(config_b)
            
        assert "Configuration mismatch" in str(excinfo.value)
        assert "prompt_version_hash" in str(excinfo.value)


# ─── Test 7: Pure PIL/NumPy Perceptual Hash Deduplication ────────────────────

class TestPerceptualHashDeduplication:
    def test_ahash_identical_and_perturbed_images(self, tmp_eval_dir):
        """Verify zero-dependency 8x8 average hash works on synthetic PIL images."""
        # 1. Base synthetic image: 100x100 gradient
        arr = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
        img1 = Image.fromarray(arr)
        
        # 2. Identical copy
        img2 = Image.fromarray(arr.copy())
        
        # 3. Slightly perturbed image (add mild noise +/- 3 pixel values)
        noise = np.random.RandomState(42).randint(-3, 4, (100, 100))
        arr_noisy = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(arr_noisy)
        
        # 4. Completely distinct image (inverted vertical gradient)
        arr_distinct = np.tile(np.linspace(255, 0, 100, dtype=np.uint8).reshape(-1, 1), (1, 100))
        img_distinct = Image.fromarray(arr_distinct)
        
        h1 = compute_average_hash(img1)
        h2 = compute_average_hash(img2)
        h_noisy = compute_average_hash(img_noisy)
        h_distinct = compute_average_hash(img_distinct)
        
        # Identical -> distance = 0
        assert hamming_distance(h1, h2) == 0
        # Noisy -> distance <= 2
        assert hamming_distance(h1, h_noisy) <= 2
        # Distinct -> distance is large (> 10)
        assert hamming_distance(h1, h_distinct) > 10

    def test_known_historical_duplicates_detected(self):
        """Verify known historical duplicate train_6993 vs train_2837 is detected with distance=0."""
        img_dir = "data/processed/images"
        p1 = os.path.join(img_dir, "train_6993.jpg")
        p2 = os.path.join(img_dir, "train_2837.jpg")
        
        if os.path.exists(p1) and os.path.exists(p2):
            h1 = compute_average_hash(p1)
            h2 = compute_average_hash(p2)
            dist = hamming_distance(h1, h2)
            assert dist == 0, f"Expected exact duplicate (dist=0), got {dist}"

    def test_ocr_text_normalization(self):
        """Verify normalize_ocr_text and similarity matching."""
        ocr1 = "Tweet Bazz\n@sweetbazz\n#WhatsApp introduces Disappearing feature!"
        ocr2 = "tweet bazz @sweetbazz whatsapp introduces disappearing feature"
        
        sim = compute_ocr_similarity(ocr1, ocr2)
        assert sim > 0.90
