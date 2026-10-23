"""
Regression and validation tests for Config C: Targeted Cultural-Irony Refinement.
"""

import json
import pytest
from src.vlm.prompts import (
    build_cultural_analysis_prompt_config_c,
    _CULTURAL_JSON_SCHEMA,
    _CLASSIFICATION_RULES,
)
from src.vlm.inference import VLMInferenceEngine


class TestConfigCPrompt:
    def test_prompt_structure_and_schema(self):
        msgs = build_cultural_analysis_prompt_config_c(
            image_path="test_image.jpg",
            ocr_text="Test OCR text",
            retrieved_context="Test Cultural Context"
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

        system_text = msgs[0]["content"][0]["text"]
        user_text = msgs[1]["content"][1]["text"]

        # Validate required clarifications
        assert "Serious cultural subject matter does not automatically imply non-humour" in system_text
        assert "political satire, irony, sarcasm, parody, and dark humour" in system_text.lower()
        assert "supporting evidence" in system_text
        assert "Do not suppress a clear comedic mechanism solely because the retrieved context describes a solemn event" in system_text
        assert "cultural familiarity, provocative language, or an unusual image alone does not establish humour" in system_text

        # Validate user content guidelines
        assert "CULTURAL CONTEXT & SATIRE/IRONY GUIDELINES:" in user_text
        assert _CULTURAL_JSON_SCHEMA in user_text
        assert _CLASSIFICATION_RULES in user_text

    def test_disagreement_cases_identification(self):
        """Programmatic verification of the exact 11 disagreement cases from A-vs-B report."""
        manifest_path = "results/evaluation_heldout_100/sample_manifest.json"
        config_a_path = "results/evaluation_heldout_100_config_a/inference_results.jsonl"
        config_b_path = "results/evaluation_heldout_100_config_b/inference_results.jsonl"

        with open(config_a_path, "r", encoding="utf-8") as f:
            results_a = {json.loads(line)["sample_index"]: json.loads(line) for line in f if line.strip()}

        with open(config_b_path, "r", encoding="utf-8") as f:
            results_b = {json.loads(line)["sample_index"]: json.loads(line) for line in f if line.strip()}

        disagreements = []
        for idx in range(100):
            a = results_a.get(idx)
            b = results_b.get(idx)
            if a and b and a.get("prediction") is not None and b.get("prediction") is not None:
                if a["prediction"] != b["prediction"]:
                    disagreements.append(idx)

        expected_11 = [0, 4, 14, 24, 60, 65, 68, 72, 73, 77, 98]
        assert disagreements == expected_11, f"Expected {expected_11}, got {disagreements}"

    def test_mock_two_stage_execution_config_c(self):
        """Verify two-stage gating and prompt builder selection for mode='cultural_c' with mock."""
        engine = VLMInferenceEngine("configs/model.yaml")
        
        # Test mock infer with stage1_result provided
        stage1_res = {
            "humorous": True,
            "prediction": "Humorous",
            "humor_probability": 0.65,
            "non_humor_probability": 0.35,
            "confidence": 0.65,
            "reason": "Stage 1 found situational irony.",
            "ocr_text": "Sample text",
        }

        # Mock _run_generation to return a valid cultural JSON
        mock_output = json.dumps({
            "humorous": True,
            "detected_text": "Sample text",
            "visual_description": "A man smiling",
            "cultural_category": "Bollywood",
            "cultural_dependency": "medium",
            "cultural_context": "Bollywood reference",
            "humor_evidence": "Satirical dialogue delivery",
            "non_humor_evidence": "None",
            "reason": "Humorous satire with clear comedic mechanism."
        })

        engine._run_generation = lambda msgs: (
            {"humorous": True, "reason": "Humorous satire"}, 0.62, 0.38, {"inference_duration_s": 0.5}
        )

        res = engine.infer(
            image_path="data/processed/images/train_5296.jpg",
            mode="cultural_c",
            ocr_text="Sample text",
            retrieved_context="PROJECT CULTURAL KNOWLEDGE: Bollywood film",
            stage1_result=stage1_res,
            two_stage=True
        )

        assert res["mode"] == "cultural_c"
        assert res["two_stage_applied"] is True
        assert res["humorous"] is True
        assert res["prediction"] == "Humorous"
        assert res["humor_probability"] == 0.62
