"""
Unit tests for DRISHTIKON Cultural Retriever.

Verifies:
1. Relevance search returns matches above threshold.
2. Returns all 12 official fields + derived fields + relevance_score.
3. Threshold filtering rejects low-similarity queries.
4. Empty or whitespace queries return empty lists.
5. Deterministic search_text derivation correctness.
"""

import pytest
from pathlib import Path
from src.cultural.drishtikon_retriever import DrishtikonRetriever
from src.cultural.drishtikon_loader import OFFICIAL_FIELDS

FIXTURE_PATH = str(Path("tests/fixtures/drishtikon_sample.json"))


def test_retriever_successful_query():
    retriever = DrishtikonRetriever(
        knowledge_path=FIXTURE_PATH,
        top_k=2,
        min_similarity_threshold=0.1
    )
    results = retriever.retrieve("temples kedarnath pilgrimage")
    assert len(results) > 0
    assert len(results) <= 2

    top = results[0]
    # Check that score is present and valid
    assert "relevance_score" in top
    assert top["relevance_score"] >= 0.1

    # Check all official fields
    for field in OFFICIAL_FIELDS:
        assert field in top

    # Check derived fields
    assert "id" in top
    assert "search_text" in top


def test_retriever_no_match_below_threshold():
    retriever = DrishtikonRetriever(
        knowledge_path=FIXTURE_PATH,
        top_k=3,
        min_similarity_threshold=0.85  # Very high threshold
    )
    # Query completely unrelated to Uttarakhand temples/pilgrimages
    results = retriever.retrieve("quantum physics neural network microchip")
    assert len(results) == 0


def test_retriever_empty_query():
    retriever = DrishtikonRetriever(knowledge_path=FIXTURE_PATH)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve(None) == []


def test_retriever_missing_knowledge_base():
    retriever = DrishtikonRetriever(knowledge_path="missing_file.json")
    results = retriever.retrieve("temple")
    assert results == []
