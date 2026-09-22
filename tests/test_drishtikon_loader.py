"""
Unit tests for DRISHTIKON Dataset Loader.

Verifies:
1. Loads valid test fixture records.
2. Validates all 12 official DRISHTIKON fields are present and preserved.
3. Validates the 2 derived processing fields ('id' and 'search_text').
4. Handles missing or invalid knowledge paths gracefully without crashing.
"""

import pytest
from pathlib import Path
from src.cultural.drishtikon_loader import DrishtikonKnowledgeBase, OFFICIAL_FIELDS

FIXTURE_PATH = Path("tests/fixtures/drishtikon_sample.json")


def test_drishtikon_loader_with_valid_fixture():
    kb = DrishtikonKnowledgeBase(str(FIXTURE_PATH))
    assert kb.is_loaded is True
    assert len(kb) == 8

    # Verify first record fields
    first = kb.get_record_by_id(0)
    assert first is not None

    # Check all 12 official fields
    for field in OFFICIAL_FIELDS:
        assert field in first, f"Official field '{field}' missing from record"
        assert isinstance(first[field], str), f"Field '{field}' should be string"

    # Check 2 derived fields
    assert "id" in first
    assert isinstance(first["id"], int)
    assert "search_text" in first
    assert isinstance(first["search_text"], str)
    assert len(first["search_text"]) > 0


def test_drishtikon_loader_missing_file_fallback():
    kb = DrishtikonKnowledgeBase("non_existent_path.json")
    assert kb.is_loaded is False
    assert len(kb) == 0
    assert kb.get_record_by_id(0) is None


def test_drishtikon_loader_empty_initialization():
    kb = DrishtikonKnowledgeBase()
    assert kb.is_loaded is False
    assert len(kb) == 0
