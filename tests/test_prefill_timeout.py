"""
Unit tests for Phase 3: Decoupling Prefill Duration from Generation Timeout.

Verifies:
1. Slow prefill (e.g. 100s) does NOT trigger a false timeout on token 1.
2. Prefill duration and decoding duration are accurately recorded.
3. Hard bound on autoregressive decoding timeout is strictly enforced.
4. Diagnostics pass prefill_duration_s and decoding_duration_s into the final result dictionary.
"""

import time
import torch
import unittest
from unittest.mock import patch, MagicMock

from src.vlm.inference import DiagnosticStoppingCriteria, VLMInferenceEngine


class TestPrefillDecodingTimeout(unittest.TestCase):
    def test_slow_prefill_does_not_abort_on_first_token(self):
        """Simulate a 100-second vision prefill; verify token 1 is NOT timed out."""
        t0 = 1000.0
        # Generation timeout is set to 10.0 seconds
        criteria = DiagnosticStoppingCriteria(
            start_time=t0,
            generation_timeout_seconds=10.0,
            initial_input_length=50
        )

        # Token 1 arrives after 100 seconds of prefill (t = 1100.0)
        input_ids_1 = torch.zeros((1, 51), dtype=torch.long)
        with patch("time.time", return_value=1100.0):
            stop_1 = criteria(input_ids_1, None)

        self.assertFalse(stop_1, "Generation should NOT stop on token 1 after slow prefill")
        self.assertFalse(criteria.timed_out)
        self.assertAlmostEqual(criteria.prefill_duration, 100.0, places=2)
        self.assertEqual(criteria.tokens_generated, 1)

    def test_normal_decoding_progress_within_timeout(self):
        """Simulate normal decoding within the 10.0s window."""
        t0 = 1000.0
        criteria = DiagnosticStoppingCriteria(
            start_time=t0,
            generation_timeout_seconds=10.0,
            initial_input_length=50
        )

        # Prefill completes at t = 1050.0 (50s prefill)
        input_ids = torch.zeros((1, 51), dtype=torch.long)
        with patch("time.time", return_value=1050.0):
            criteria(input_ids, None)

        # Decoding generates 30 tokens over 3 seconds (t = 1053.0)
        input_ids_30 = torch.zeros((1, 80), dtype=torch.long)
        with patch("time.time", return_value=1053.0):
            stop = criteria(input_ids_30, None)

        self.assertFalse(stop)
        self.assertFalse(criteria.timed_out)
        self.assertEqual(criteria.tokens_generated, 30)

    def test_hard_bound_enforced_when_decoding_exceeds_timeout(self):
        """Verify hard bound is enforced when token decoding itself exceeds the limit."""
        t0 = 1000.0
        criteria = DiagnosticStoppingCriteria(
            start_time=t0,
            generation_timeout_seconds=5.0,
            initial_input_length=50
        )

        # Step 1: Prefill ends at t = 1020.0
        input_ids_1 = torch.zeros((1, 51), dtype=torch.long)
        with patch("time.time", return_value=1020.0):
            criteria(input_ids_1, None)

        # Step 2: Decoding reaches t = 1026.0 (6 seconds of decoding > 5.0s timeout)
        input_ids_2 = torch.zeros((1, 60), dtype=torch.long)
        with patch("time.time", return_value=1026.0):
            stop = criteria(input_ids_2, None)

        self.assertTrue(stop, "Generation MUST stop when decoding exceeds generation_timeout_seconds")
        self.assertTrue(criteria.timed_out)

    def test_diagnostics_include_prefill_and_decoding_durations(self):
        """Verify _build_result propagates prefill_duration_s and decoding_duration_s."""
        engine = VLMInferenceEngine("configs/model.yaml")
        diagnostics = {
            "timed_out": False,
            "hit_max_new_tokens": False,
            "generated_token_count": 150,
            "raw_response": '{"humorous": false}',
            "inference_duration_s": 120.5,
            "prefill_duration_s": 95.2,
            "decoding_duration_s": 25.3,
        }
        parsed = {"humorous": False, "reason": "Serious", "prediction": "Not Humorous"}
        res = engine._build_result(parsed, humor_prob=0.1, non_humor_prob=0.9, mode="general", diagnostics=diagnostics)

        self.assertEqual(res["prefill_duration_s"], 95.2)
        self.assertEqual(res["decoding_duration_s"], 25.3)
        self.assertEqual(res["inference_duration_s"], 120.5)


if __name__ == "__main__":
    unittest.main()
