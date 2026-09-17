import unittest
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
