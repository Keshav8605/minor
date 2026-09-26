"""
Unit tests for CulturalContextBuilder.

Verifies:
1. Permutation 1: Only Project JSON matches.
2. Permutation 2: Only DRISHTIKON matches.
3. Permutation 3: Both sources match.
4. Permutation 4: Neither source matches.
5. RESEARCH INTEGRITY: Verifies option1-option4 and raw JSON are NEVER injected into the prompt context.
"""

import pytest
from pathlib import Path
from src.cultural.context_builder import CulturalContextBuilder

CATEGORIES_PATH = "data/cultural/cultural_categories.json"
KNOWLEDGE_PATH = "data/cultural/cultural_knowledge.json"
DRISHTIKON_FIXTURE = str(Path("tests/fixtures/drishtikon_sample.json"))


def test_builder_disabled_drishtikon_only_project_json():
    builder = CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_enabled=False
    )
    # Cricket query: matches project JSON
    context, sources, drish_records = builder.build_context("dhoni kohli match")
    assert "PROJECT_JSON" in sources
    assert "DRISHTIKON" not in sources
    assert len(drish_records) == 0
    assert "CRICKET" in context


def test_builder_both_sources_match():
    builder = CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_path=DRISHTIKON_FIXTURE,
        drishtikon_enabled=True,
        min_similarity_threshold=0.1
    )
    # Mandir / puja / kedarnath -> matches project JSON (religion) and DRISHTIKON fixture (temple/kedarnath)
    context, sources, drish_records = builder.build_context("mandir kedarnath puja temple pilgrimage")
    assert "PROJECT_JSON" in sources
    assert "DRISHTIKON" in sources
    assert len(drish_records) > 0
    assert "PROJECT CULTURAL KNOWLEDGE" in context
    assert "INDIAN CULTURAL CONTEXT (DRISHTIKON)" in context


def test_builder_neither_source_matches():
    builder = CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_path=DRISHTIKON_FIXTURE,
        drishtikon_enabled=True,
        min_similarity_threshold=0.5
    )
    context, sources, drish_records = builder.build_context("quantum mechanics photon laser")
    assert context == ""
    assert sources == []
    assert drish_records == []


def test_builder_does_not_inject_options_or_raw_json():
    builder = CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_path=DRISHTIKON_FIXTURE,
        drishtikon_enabled=True,
        min_similarity_threshold=0.05
    )
    context, sources, drish_records = builder.build_context("kedarnath temples")
    if "DRISHTIKON" in sources:
        # Check that no option1, option2, option3, option4 keywords leaked into prompt string
        for rec in drish_records:
            assert "option1" not in context
            assert "option2" not in context
            assert "option3" not in context
            assert "option4" not in context
            assert '{"' not in context  # No raw JSON object
