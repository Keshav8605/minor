"""
Unit tests for the five targeted reliability fixes:
1. Truncated JSON recovery and conservative parsing
2. Generation diagnostics and timeout propagation
3. Stage-2 fallback to valid Stage-1 (and refusal when Stage-1 is invalid)
4. Evaluation metadata and failure accounting on an 11-sample run without launching VLM
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.vlm.output_parser import parse_json_response, normalize_result_schema
from src.vlm.inference import VLMInferenceEngine
from scripts.run_controlled_evaluation import run_controlled_experiment


class TestTruncatedJsonRecovery(unittest.TestCase):
    def test_truncated_json_with_valid_boolean_is_recovered(self):
        """Truncated JSON with an unclosed object is recovered if required fields exist."""
        truncated_raw = '{"humorous": true, "reason": "Funny contrast between image and text", "confidence": 0.85'
        parsed = parse_json_response(truncated_raw)
        self.assertIsNone(parsed.get("error"))
        self.assertTrue(parsed.get("humorous"))
        self.assertEqual(parsed.get("prediction"), "Humorous")
        self.assertIn("Funny contrast", parsed.get("reason"))

    def test_truncated_json_with_trailing_unclosed_string(self):
        """Truncated JSON cut off mid-string in reasoning recovers cleanly."""
        truncated_raw = '{"humorous": false, "reason": "This is a serious public announcement that has no punch'
        parsed = parse_json_response(truncated_raw)
        self.assertIsNone(parsed.get("error"))
        self.assertFalse(parsed.get("humorous"))
        self.assertEqual(parsed.get("prediction"), "Not Humorous")
        self.assertIn("serious public announcement", parsed.get("reason"))

    def test_missing_humorous_field_does_not_assume_non_humorous(self):
        """If JSON is truncated before 'humorous' boolean is declared, NEVER assume non-humorous."""
        raw = '{"detected_text": "Breaking news", "reason": "Incomplete output'
        parsed = parse_json_response(raw)
        self.assertIsNotNone(parsed.get("error"))
        self.assertEqual(parsed.get("prediction"), "Unavailable")
        self.assertIn("Inference failed or output was unparseable", parsed.get("reasoning"))

    def test_completely_malformed_output_returns_unavailable_not_false(self):
        """Completely malformed text must return error with Unavailable prediction, not humorous=False."""
        raw = "Sorry, I am unable to process this request due to an internal timeout."
        parsed = parse_json_response(raw)
        self.assertIsNotNone(parsed.get("error"))
        self.assertIsNone(parsed.get("humorous"))
        self.assertEqual(parsed.get("prediction"), "Unavailable")


class TestDiagnosticsAndTimeoutPropagation(unittest.TestCase):
    def setUp(self):
        self.engine = VLMInferenceEngine("configs/model.yaml")

    def test_build_result_propagates_diagnostics(self):
        """Diagnostics from generation are attached to the final result dictionary."""
        diagnostics = {
            "timed_out": True,
            "hit_max_new_tokens": True,
            "generated_token_count": 256,
            "raw_response": '{"humorous": true}',
            "inference_duration_s": 42.5,
        }
        parsed = {"humorous": True, "reason": "Test reason", "prediction": "Humorous"}
        res = self.engine._build_result(
            parsed, humor_prob=0.88, non_humor_prob=0.12, mode="general", diagnostics=diagnostics
        )
        self.assertTrue(res["timed_out"])
        self.assertTrue(res["hit_max_new_tokens"])
        self.assertEqual(res["generated_token_count"], 256)
        self.assertEqual(res["raw_response"], '{"humorous": true}')
        self.assertEqual(res["inference_duration_s"], 42.5)

    def test_unparseable_output_builds_unavailable_prediction(self):
        """When output parsing fails, _build_result preserves prediction='Unavailable'."""
        parsed = {
            "humorous": None,
            "prediction": "Unavailable",
            "reason": "Inference failed or output was unparseable.",
            "error": "Failed to extract valid JSON",
        }
        res = self.engine._build_result(parsed, humor_prob=None, non_humor_prob=None, mode="general")
        self.assertIsNone(res["humorous"])
        self.assertEqual(res["prediction"], "Unavailable")
        self.assertIsNotNone(res["error"])


class TestStage2FallbackToStage1(unittest.TestCase):
    def setUp(self):
        self.engine = VLMInferenceEngine("configs/model.yaml")
        self.test_image = "data/processed/images/train_5266.jpg"

    def test_stage2_failure_falls_back_to_valid_stage1(self):
        """When Stage 2 parsing fails or times out, valid Stage 1 is used with explicit flags."""
        stage1_res = {
            "humorous": True,
            "prediction": "Humorous",
            "humor_probability": 0.85,
            "non_humor_probability": 0.15,
            "reason": "Stage-1 joke",
            "detected_text": "Sample joke",
        }

        # Stage 2 generation returns unparseable error
        stage2_error_parsed = {
            "humorous": None,
            "prediction": "Unavailable",
            "reason": "Inference failed or output was unparseable.",
            "error": "JSONDecodeError: Unterminated string",
        }
        diagnostics = {
            "timed_out": True,
            "hit_max_new_tokens": False,
            "generated_token_count": 180,
            "raw_response": '{"humorous": ...',
            "inference_duration_s": 120.0,
        }
        self.engine._run_generation = MagicMock(return_value=(stage2_error_parsed, None, None, diagnostics))

        res = self.engine.infer(
            image_path=self.test_image,
            mode="cultural",
            ocr_text="Sample joke",
            retrieved_context="Some valid cultural context",
            stage1_result=stage1_res,
            two_stage=True,
        )

        self.assertTrue(res["humorous"])
        self.assertTrue(res["stage2_failed"])
        self.assertTrue(res["stage2_fallback_to_stage1"])
        self.assertEqual(res["stage2_error"], "JSONDecodeError: Unterminated string")
        self.assertTrue(res["timed_out"])
        self.assertEqual(res["generated_token_count"], 180)

    def test_stage2_failure_without_valid_stage1_does_not_fallback(self):
        """When Stage 2 fails and Stage 1 was ALSO invalid/missing, return error prediction."""
        # Stage 1 was invalid (e.g. failed general inference)
        stage1_res = {
            "humorous": None,
            "prediction": "Unavailable",
            "humor_probability": None,
            "reason": "Stage 1 failed",
        }

        stage2_error_parsed = {
            "humorous": None,
            "prediction": "Unavailable",
            "reason": "Inference failed or output was unparseable.",
            "error": "Failed to parse JSON",
        }
        self.engine._run_generation = MagicMock(return_value=(stage2_error_parsed, None, None, {}))

        res = self.engine.infer(
            image_path=self.test_image,
            mode="cultural",
            ocr_text="nibba nibbi dating",
            retrieved_context="RELATIONSHIPS: Indian relationship memes",
            stage1_result=stage1_res,
            two_stage=True,
        )

        self.assertTrue(res["stage2_failed"])
        self.assertFalse(res.get("stage2_fallback_to_stage1", False))
        self.assertIsNone(res["humorous"])
        self.assertEqual(res["prediction"], "Unavailable")


class TestEvaluationMetadataAndFailureAccounting(unittest.TestCase):
    def test_evaluation_summary_on_partial_11_sample_run(self):
        """Test evaluation summary computation on an 11-sample run using mock inference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            call_count = {"general": 0, "cultural": 0}

            def mock_infer(image_path, condition, ocr_text, context, sources):
                call_count[condition] += 1
                sample_num = call_count[condition]
                
                # Introduce 1 failure in general mode on sample 3
                if condition == "general" and sample_num == 3:
                    return {
                        "humorous": None,
                        "prediction": "Unavailable",
                        "humor_probability": None,
                        "non_humor_probability": None,
                        "reason": "Simulated parsing failure",
                        "error": "JSONDecodeError: invalid syntax",
                        "timed_out": False,
                        "hit_max_new_tokens": False,
                        "generated_token_count": 50,
                    }

                # Introduce 1 Stage-2 fallback in cultural mode on sample 7
                if condition == "cultural" and sample_num == 7:
                    return {
                        "humorous": True,
                        "prediction": "Humorous",
                        "humor_probability": 0.70,
                        "non_humor_probability": 0.30,
                        "reason": "Fell back to Stage-1 assessment",
                        "stage2_failed": True,
                        "stage2_fallback_to_stage1": True,
                        "stage2_error": "Timeout in Stage 2",
                        "timed_out": True,
                        "cultural_category": "relationships",
                    }

                # Standard mock predictions
                is_humorous = (sample_num % 2 == 1)
                return {
                    "humorous": is_humorous,
                    "prediction": "Humorous" if is_humorous else "Not Humorous",
                    "humor_probability": 0.80 if is_humorous else 0.20,
                    "non_humor_probability": 0.20 if is_humorous else 0.80,
                    "reason": "Mock reason",
                    "timed_out": False,
                    "cultural_category": "pop_culture" if condition == "cultural" else None,
                }

            summary = run_controlled_experiment(
                test_csv_path="data/processed/test.csv",
                image_dir="data/processed/images",
                results_dir=temp_dir,
                sample_size=20,
                max_samples=11,
                infer_fn=mock_infer,
            )

            self.assertIsNotNone(summary)
            # Evaluated samples count must match the 11 evaluated samples, NOT the full manifest size (20)
            self.assertEqual(summary["evaluated_samples"], 11)
            self.assertEqual(summary["total_samples"], 11)
            self.assertEqual(summary["manifest_total_samples"], 20)

            # Ground truth counts must be derived from the 11 evaluated samples
            self.assertEqual(
                summary["humorous_ground_truth"] + summary["non_humorous_ground_truth"],
                11
            )
            self.assertEqual(summary["manifest_humorous_samples"], 10)
            self.assertEqual(summary["manifest_non_humorous_samples"], 10)

            # Reliability & Failure counts
            reliability = summary["inference_reliability"]
            self.assertEqual(reliability["total_attempts"], 22)
            self.assertEqual(reliability["general_attempts"], 11)
            self.assertEqual(reliability["cultural_attempts"], 11)
            self.assertEqual(reliability["general_failures"], 1)
            self.assertEqual(reliability["general_evaluated"], 10)  # 1 failure excluded
            self.assertEqual(reliability["cultural_evaluated"], 11)  # 1 fallback included
            self.assertEqual(reliability["cultural_stage2_fallbacks"], 1)

            # Metrics should be computed on valid inferences only
            gen_metrics = summary["general_mode_metrics"]
            self.assertEqual(gen_metrics["evaluated_inferences"], 10)
            self.assertIsNotNone(gen_metrics["accuracy"])
            self.assertIsNotNone(gen_metrics["humor_f1"])

            cult_metrics = summary["cultural_mode_metrics"]
            self.assertEqual(cult_metrics["evaluated_inferences"], 11)
            self.assertIsNotNone(cult_metrics["accuracy"])


if __name__ == "__main__":
    unittest.main()
