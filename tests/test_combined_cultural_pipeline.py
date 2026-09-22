"""
Integration tests for the combined cultural humor detection pipeline.

Verifies:
1. General Mode Baseline Purity: Never contains DRISHTIKON or cultural context.
2. Cultural-Aware Mode: Can include DRISHTIKON context when enabled.
3. Multi-Meme Isolation: Per-image context isolation is preserved.
4. Evaluation Output Formatting: to_evaluation_dict produces valid research records.
"""

import pytest
from pathlib import Path
from src.cultural.context_builder import CulturalContextBuilder
from src.vlm.inference import VLMInferenceEngine

CATEGORIES_PATH = "data/cultural/cultural_categories.json"
KNOWLEDGE_PATH = "data/cultural/cultural_knowledge.json"
DRISHTIKON_FIXTURE = str(Path("tests/fixtures/drishtikon_sample.json"))


def test_general_mode_baseline_purity():
    """Verifies that General mode outputs do not include cultural retrieval fields."""
    engine = VLMInferenceEngine()
    fake_parsed = {
        "humorous": True,
        "detected_text": "Sample meme text",
        "visual_description": "A person laughing",
        "reason": "Relatable situational humor"
    }
    result = engine._build_result(
        fake_parsed,
        humor_prob=0.85,
        non_humor_prob=0.15,
        mode="general",
        cultural_sources=[]
    )
    assert result["mode"] == "general"
    assert result["cultural_sources"] == []
    assert result["cultural_category"] == "Not analyzed (General mode)"
    assert result["cultural_dependency"] == "Not analyzed (General mode)"
    assert result["cultural_context"] == "Not analyzed (General mode)"
    assert result["humor_probability"] == 0.85
    assert result["humorous"] is True


def test_cultural_mode_result_structure():
    """Verifies that Cultural-Aware mode preserves cultural fields and sources."""
    engine = VLMInferenceEngine()
    fake_parsed = {
        "humorous": True,
        "detected_text": "Kedarnath mandir yatra",
        "cultural_category": "religion",
        "cultural_dependency": "high",
        "cultural_context": "Refers to holy pilgrimage in Uttarakhand",
        "reason": "Humorous depiction of arduous yatra"
    }
    result = engine._build_result(
        fake_parsed,
        humor_prob=0.78,
        non_humor_prob=0.22,
        mode="cultural",
        cultural_sources=["PROJECT_JSON", "DRISHTIKON"]
    )
    assert result["mode"] == "cultural"
    assert "DRISHTIKON" in result["cultural_sources"]
    assert "PROJECT_JSON" in result["cultural_sources"]
    assert result["cultural_category"] == "religion"
    assert result["cultural_dependency"] == "high"

    # Test standardized evaluation dictionary export
    eval_dict = VLMInferenceEngine.to_evaluation_dict(result, meme_id="test_meme_001")
    assert eval_dict["meme_id"] == "test_meme_001"
    assert eval_dict["mode"] == "cultural"
    assert eval_dict["humor_prediction"] == "Humorous"
    assert eval_dict["model_probability"] == 0.78
    assert "DRISHTIKON" in eval_dict["cultural_sources"]


def test_multi_image_context_isolation():
    """Verifies that cultural context is built independently per image input string."""
    builder = CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_path=DRISHTIKON_FIXTURE,
        drishtikon_enabled=True,
        min_similarity_threshold=0.1
    )
    # Image 1 has cricket keywords
    ctx1, sources1, _ = builder.build_context("dhoni kohli match ipl")
    # Image 2 has no cultural keywords
    ctx2, sources2, _ = builder.build_context("random tech code keyboard")

    assert "PROJECT_JSON" in sources1
    assert len(sources2) == 0
    assert ctx2 == ""
    assert ctx1 != ctx2
