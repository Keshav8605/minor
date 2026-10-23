"""
Tests for Phase 2: Hinglish Relationship Context Enrichment.

Verifies:
1. data/cultural/cultural_knowledge.json remains valid JSON and preserves all keys.
2. The 'relationships' entry clarifies 'milna / nahi mil sakte' as meeting or seeing a partner.
3. ContextBuilder retrieves this enriched context for relationship queries (including train_5890.jpg).
4. ContextBuilder does NOT contaminate unrelated queries with relationship context.
"""

import json
import pytest
from pathlib import Path
from src.cultural.context_builder import CulturalContextBuilder


def test_cultural_knowledge_json_schema():
    """Verify cultural_knowledge.json is valid and contains expected categories."""
    path = Path("data/cultural/cultural_knowledge.json")
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "relationships" in data
    assert "family" in data
    assert "workplace" in data
    rel_text = data["relationships"]
    assert "milna / nahi mil sakte" in rel_text
    assert "meeting or seeing a partner" in rel_text
    assert "lockdowns or curfews" in rel_text


def test_relationship_context_retrieval():
    """Verify relationship context is retrieved for OCR text containing relationship markers."""
    builder = CulturalContextBuilder(
        categories_path="data/cultural/cultural_categories.json",
        knowledge_path="data/cultural/cultural_knowledge.json",
        drishtikon_enabled=False
    )

    ocr_5890 = (
        "PM Modi Speech Live:\r\n"
        "Narendra Modi Says India Will Be Under Complete Lockdown From Midnight For 21 Days\r\n"
        "Nibbi *Nibba\r\n"
        "BABY HUM OR 21 DIN NAHI MIL SAKTE\r\n"
        "Indya Speak"
    )

    context, sources, _ = builder.build_context(ocr_5890)
    assert "PROJECT_JSON" in sources
    assert "- RELATIONSHIPS:" in context
    assert "milna / nahi mil sakte" in context
    assert "meeting or seeing a partner" in context


def test_unrelated_queries_not_contaminated():
    """Verify relationship context does not leak into unrelated queries."""
    builder = CulturalContextBuilder(
        categories_path="data/cultural/cultural_categories.json",
        knowledge_path="data/cultural/cultural_knowledge.json",
        drishtikon_enabled=False
    )

    unrelated_queries = [
        "hot Indian milfs",
        "I DONT KNOW WHO YOU ARE AB KI BAAR MODI SARKAAR buffedd",
        "engineering college hostel semester exam backlog topper",
        "chai samosa pani puri street food",
        "cricket match world cup stadium dhoni virat kohli",
    ]

    for q in unrelated_queries:
        context, sources, _ = builder.build_context(q)
        assert "- RELATIONSHIPS:" not in context, f"Query '{q}' should not contain relationships context"
