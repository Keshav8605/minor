"""
Unit tests for Stage-1 short-circuit gating in VLMInferenceEngine.
Verifies that:
1. Stage-1 non-humorous with NO cultural evidence short-circuits.
2. Stage-1 non-humorous WITH meaningful cultural markers & valid context runs Stage 2.
3. infer_multi exhibits the same gated behavior.
4. General mode is unaffected.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.vlm.inference import VLMInferenceEngine


TEST_IMAGE = "data/processed/images/train_5266.jpg"


class TestStage1ShortCircuitGating(unittest.TestCase):
    def setUp(self):
        self.engine = VLMInferenceEngine("configs/model.yaml")

    def test_infer_non_humorous_without_cultural_markers_short_circuits(self):
        """When Stage-1 is non-humorous and no cultural markers exist, fast short-circuit."""
        stage1_res = {
            "humorous": False,
            "prediction": "Not Humorous",
            "humor_probability": 0.148,
            "non_humor_probability": 0.852,
            "reason": "Not funny",
            "detected_text": "hot Indian milfs",
        }

        # Mock _run_generation so if Stage 2 is accidentally called, we catch it
        self.engine._run_generation = MagicMock(side_effect=AssertionError("Stage 2 should not be called!"))

        res = self.engine.infer(
            image_path=TEST_IMAGE,
            mode="cultural",
            ocr_text="hot Indian milfs",
            retrieved_context="",  # No cultural context
            stage1_result=stage1_res,
            two_stage=True
        )

        self.assertFalse(res["humorous"])
        self.assertTrue(res["two_stage_applied"])
        self.assertFalse(res["context_passed_to_qwen"])
        self.assertFalse(res["stage1_humorous"])
        self.engine._run_generation.assert_not_called()

    def test_infer_non_humorous_with_meaningful_cultural_markers_runs_stage2(self):
        """When Stage-1 is non-humorous BUT meaningful cultural categories and context exist, run Stage 2."""
        stage1_res = {
            "humorous": False,
            "prediction": "Not Humorous",
            "humor_probability": 0.3039,
            "non_humor_probability": 0.6961,
            "reason": "News broadcast",
            "detected_text": "PM Modi Speech Live: Nibbi Nibba BABY HUM OR 21 DIN NAHI MIL SAKTE",
        }

        # Stage 2 generation mock returns humorous=True based on cultural understanding
        stage2_parsed = {
            "humorous": True,
            "detected_text": "Nibba Nibbi meme",
            "reason": "Comedic teenage melodrama during lockdown",
            "cultural_category": "relationships",
            "cultural_dependency": "high",
        }
        self.engine._run_generation = MagicMock(return_value=(stage2_parsed, 0.7250, 0.2750))

        cultural_context = "EXTERNAL CULTURAL CONTEXT:\n- RELATIONSHIPS: Nibba-nibbi teenage romance tropes"

        res = self.engine.infer(
            image_path=TEST_IMAGE,
            mode="cultural",
            ocr_text="Nibbi *Nibba BABY HUM OR 21 DIN NAHI MIL SAKTE",
            retrieved_context=cultural_context,
            cultural_sources=["PROJECT_JSON"],
            stage1_result=stage1_res,
            two_stage=True
        )

        # Stage 2 should have been called and changed prediction to humorous
        self.assertTrue(res["humorous"])
        self.assertTrue(res["two_stage_applied"])
        self.assertTrue(res["context_passed_to_qwen"])
        self.assertFalse(res["stage1_humorous"])  # Preserves that Stage 1 was originally False
        self.assertEqual(res["humor_probability"], 0.7250)
        self.engine._run_generation.assert_called_once()

    def test_infer_humorous_without_cultural_context_reuses_stage1(self):
        """When Stage-1 is humorous but no cultural context is needed, reuse Stage-1 result."""
        stage1_res = {
            "humorous": True,
            "prediction": "Humorous",
            "humor_probability": 0.7930,
            "non_humor_probability": 0.2070,
            "reason": "Cyanide and happiness comic",
            "detected_text": "Do you support Reddit Admin",
        }

        self.engine._run_generation = MagicMock(side_effect=AssertionError("Stage 2 should not run without context"))

        res = self.engine.infer(
            image_path=TEST_IMAGE,
            mode="cultural",
            ocr_text="Do you support Reddit Admin",
            retrieved_context="",  # No cultural context
            stage1_result=stage1_res,
            two_stage=True
        )

        self.assertTrue(res["humorous"])
        self.assertTrue(res["two_stage_applied"])
        self.assertFalse(res["context_passed_to_qwen"])
        self.assertTrue(res["stage1_humorous"])
        self.engine._run_generation.assert_not_called()

    def test_infer_multi_non_humorous_gating(self):
        """infer_multi also respects gated short-circuiting."""
        stage1_res = {
            "humorous": False,
            "prediction": "Not Humorous",
            "humor_probability": 0.2,
            "non_humor_probability": 0.8,
            "reason": "No humor",
            "detected_text": "Plain text",
        }

        # Case A: No cultural markers -> short-circuits
        self.engine._run_generation = MagicMock(side_effect=AssertionError("Should not call Stage 2"))
        res_a = self.engine.infer_multi(
            image_paths=[TEST_IMAGE],
            mode="cultural",
            ocr_text="Just normal text",
            retrieved_context="",
            stage1_result=stage1_res,
            two_stage=True
        )
        self.assertFalse(res_a["humorous"])
        self.assertFalse(res_a["context_passed_to_qwen"])
        self.engine._run_generation.assert_not_called()

        # Case B: Meaningful cultural markers -> runs Stage 2
        stage2_parsed = {
            "humorous": True,
            "reason": "Cultural joke understood",
            "cultural_category": "cricket",
            "cultural_dependency": "medium"
        }
        self.engine._run_generation = MagicMock(return_value=(stage2_parsed, 0.65, 0.35))
        res_b = self.engine.infer_multi(
            image_paths=[TEST_IMAGE],
            mode="cultural",
            ocr_text="Dhoni finish in style IPL match",
            retrieved_context="EXTERNAL CULTURAL CONTEXT:\n- CRICKET: Dhoni helicopter shot",
            cultural_sources=["PROJECT_JSON"],
            stage1_result=stage1_res,
            two_stage=True
        )
        self.assertTrue(res_b["humorous"])
        self.assertTrue(res_b["context_passed_to_qwen"])
        self.engine._run_generation.assert_called_once()
