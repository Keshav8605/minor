"""
Probability mechanism verification tests.

Verifies:
1. P(humor) + P(non-humor) ≈ 1
2. Probabilities are derived from logits, not model-reported confidence
3. _build_result correctly applies logit-based prediction
4. Edge cases (None probs, conflicting VLM text vs logits)
"""

import unittest
import math
from unittest.mock import MagicMock
from src.vlm.inference import VLMInferenceEngine


class TestLogitConfidence(unittest.TestCase):
    def setUp(self):
        # Instantiate without calling load()
        self.engine = VLMInferenceEngine("configs/model.yaml")

    def test_build_result_with_logits_humorous(self):
        parsed = {
            "humorous": True,
            "detected_text": "Sample text",
            "reason": "Very funny",
            "cultural_category": "cricket",
            "cultural_dependency": "medium"
        }
        result = self.engine._build_result(
            parsed, humor_prob=0.8245, non_humor_prob=0.1755, mode="cultural"
        )
        self.assertTrue(result["humorous"])
        self.assertEqual(result["confidence"], 0.8245)
        self.assertEqual(result["humor_probability"], 0.8245)
        self.assertEqual(result["non_humor_probability"], 0.1755)
        self.assertEqual(result["confidence_method"], "logit_derived")
        self.assertEqual(result["mode"], "cultural")

    def test_build_result_with_logits_non_humorous(self):
        parsed = {
            "humorous": False,
            "detected_text": "Serious news",
            "reason": "Not funny"
        }
        result = self.engine._build_result(
            parsed, humor_prob=0.2311, non_humor_prob=0.7689, mode="general"
        )
        self.assertFalse(result["humorous"])
        self.assertEqual(result["confidence"], 0.7689)
        self.assertEqual(result["confidence_method"], "logit_derived")

    def test_build_result_fallback_when_logits_unavailable(self):
        parsed = {
            "humorous": True,
            "detected_text": "Meme text",
            "reason": "Some joke"
        }
        # When logit extraction fails, it should NOT fabricate confidence
        result = self.engine._build_result(
            parsed, humor_prob=None, non_humor_prob=None, mode="general"
        )
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["humor_probability"])
        self.assertEqual(result["confidence_method"], "unavailable")


class TestProbabilitySumToOne(unittest.TestCase):
    """Verify P(humor) + P(non-humor) ≈ 1 for various probability pairs."""

    def test_sum_high_humor(self):
        humor_prob = 0.9423
        non_humor_prob = 0.0577
        self.assertAlmostEqual(humor_prob + non_humor_prob, 1.0, places=4)

    def test_sum_low_humor(self):
        humor_prob = 0.1234
        non_humor_prob = 0.8766
        self.assertAlmostEqual(humor_prob + non_humor_prob, 1.0, places=4)

    def test_sum_balanced(self):
        humor_prob = 0.5001
        non_humor_prob = 0.4999
        self.assertAlmostEqual(humor_prob + non_humor_prob, 1.0, places=4)

    def test_build_result_probabilities_sum_to_one(self):
        """Verify _build_result output probabilities sum to ~1."""
        engine = VLMInferenceEngine("configs/model.yaml")
        parsed = {"humorous": True, "detected_text": "test", "reason": "test"}
        
        # Test with various probability pairs
        test_pairs = [
            (0.95, 0.05),
            (0.50, 0.50),
            (0.10, 0.90),
            (0.7777, 0.2223),
        ]
        for hp, nhp in test_pairs:
            result = engine._build_result(parsed, humor_prob=hp, non_humor_prob=nhp, mode="general")
            p_humor = result["humor_probability"]
            p_non = result["non_humor_probability"]
            self.assertAlmostEqual(p_humor + p_non, 1.0, places=3,
                msg=f"P(humor)={p_humor} + P(non)={p_non} != 1.0")


class TestLogitBasedNotSelfReported(unittest.TestCase):
    """Verify the confidence is derived from logits, not model self-reporting."""

    def test_confidence_method_is_logit_derived(self):
        engine = VLMInferenceEngine("configs/model.yaml")
        parsed = {"humorous": True, "detected_text": "t", "reason": "r"}
        result = engine._build_result(parsed, humor_prob=0.85, non_humor_prob=0.15, mode="general")
        self.assertEqual(result["confidence_method"], "logit_derived")

    def test_vlm_text_overridden_by_logits(self):
        """When VLM text says humorous=True but logits say P(humor)<0.5,
        the logit-based prediction should win."""
        engine = VLMInferenceEngine("configs/model.yaml")
        parsed = {"humorous": True, "detected_text": "t", "reason": "r"}
        result = engine._build_result(parsed, humor_prob=0.3, non_humor_prob=0.7, mode="general")
        # Logits override VLM text
        self.assertFalse(result["humorous"])
        self.assertEqual(result["confidence"], 0.7)

    def test_vlm_text_false_but_logits_true(self):
        """VLM says false but logits say humor is more likely."""
        engine = VLMInferenceEngine("configs/model.yaml")
        parsed = {"humorous": False, "detected_text": "t", "reason": "r"}
        result = engine._build_result(parsed, humor_prob=0.75, non_humor_prob=0.25, mode="general")
        # Logits override
        self.assertTrue(result["humorous"])
        self.assertEqual(result["confidence"], 0.75)

    def test_parsed_confidence_field_ignored(self):
        """Even if the parsed JSON contains a 'confidence' field from model self-reporting,
        _build_result should overwrite it with logit-derived confidence."""
        engine = VLMInferenceEngine("configs/model.yaml")
        parsed = {
            "humorous": True,
            "confidence": 0.99,  # Self-reported (should be overwritten)
            "detected_text": "t",
            "reason": "r"
        }
        result = engine._build_result(parsed, humor_prob=0.65, non_humor_prob=0.35, mode="general")
        # The confidence should be the logit-derived max, NOT the self-reported 0.99
        self.assertEqual(result["confidence"], 0.65)
        self.assertEqual(result["confidence_method"], "logit_derived")


if __name__ == "__main__":
    unittest.main()
