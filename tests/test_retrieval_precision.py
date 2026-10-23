"""
Tests for DRISHTIKON Retrieval Precision and Query Sanitization.

Verifies Phase 1 improvements:
1. configs/cultural.yaml defines min_similarity_threshold == 0.25.
2. Default thresholds in DrishtikonRetriever and CulturalContextBuilder are 0.25.
3. Query sanitization removes meme prefixes, credit handles (@user, ig/handle, backup: @...),
   and isolated numbers, while preserving cultural numbers (370, 144, 420) and dates (15 August).
4. Spurious single-word collisions ('earning', 'courage', 'hot') are rejected below 0.25.
5. Legitimate cultural entity queries remain retrievable.
"""

import pytest
import yaml
from pathlib import Path
from src.cultural.context_builder import sanitize_retrieval_query, CulturalContextBuilder
from src.cultural.drishtikon_retriever import DrishtikonRetriever

FIXTURE_PATH = str(Path("tests/fixtures/drishtikon_sample.json"))


def test_cultural_config_threshold():
    """Verify that configs/cultural.yaml sets min_similarity_threshold to 0.25."""
    with open("configs/cultural.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config.get("min_similarity_threshold") == 0.25


def test_default_retriever_threshold():
    """Verify default threshold in DrishtikonRetriever is 0.25."""
    retriever = DrishtikonRetriever(knowledge_path=FIXTURE_PATH)
    assert retriever.min_similarity_threshold == 0.25


def test_query_sanitization_removes_handles_and_prefixes():
    """Verify sanitization strips social handles and boilerplate prefixes."""
    raw = "Me and the boys after earning 10,000 for reporting fake Whatsapp videos @memeguy ig/dankmemes"
    sanitized = sanitize_retrieval_query(raw)
    assert "me and the boys" not in sanitized.lower()
    assert "10,000" not in sanitized
    assert "@memeguy" not in sanitized
    assert "ig/dankmemes" not in sanitized
    assert "reporting fake whatsapp videos" in sanitized.lower()


def test_query_sanitization_preserves_cultural_numbers_and_dates():
    """Verify sanitization preserves legal/cultural sections and dates."""
    raw = "Protest under Section 144 and Article 370 on 15 August 420 fraud"
    sanitized = sanitize_retrieval_query(raw)
    assert "Section 144" in sanitized
    assert "Article 370" in sanitized
    assert "15 August" in sanitized
    assert "420" in sanitized


def test_spurious_noise_queries_rejected_at_threshold():
    """Verify that spurious single-word collisions (earning, courage, hot) are rejected."""
    retriever = DrishtikonRetriever(
        knowledge_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
        min_similarity_threshold=0.25
    )
    if not retriever.kb.is_loaded:
        pytest.skip("Full DRISHTIKON knowledge base not available in test environment.")

    # Spurious noise from evaluation memes
    noise_queries = [
        "earning for reporting fake Whatsapp videos amc",  # train_1611.jpg
        "THIS is not COURAGE. THIS is COURAGE.",          # train_5088.jpg
        "hot Indian milfs",                               # train_5266.jpg
        "Bore hore hoge chalo thoda Hans lo Quarantine Day", # train_4199.jpg
    ]

    for q in noise_queries:
        matches = retriever.retrieve(q)
        assert len(matches) == 0, f"Query '{q}' should have 0 matches at 0.25 threshold, got {len(matches)}"


def test_legitimate_cultural_entities_retrieved_at_threshold():
    """Verify that specific cultural entity queries remain retrievable at 0.25."""
    retriever = DrishtikonRetriever(
        knowledge_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
        min_similarity_threshold=0.25
    )
    if not retriever.kb.is_loaded:
        pytest.skip("Full DRISHTIKON knowledge base not available in test environment.")

    entity_queries = [
        "kedarnath temple uttarakhand pilgrimage",
        "badami rock cut cave temples karnataka",
        "shah rukh khan king of bollywood actor",
    ]

    for q in entity_queries:
        matches = retriever.retrieve(q)
        assert len(matches) > 0, f"Query '{q}' should have matched DRISHTIKON at 0.25 threshold"
        assert matches[0]["relevance_score"] >= 0.25
