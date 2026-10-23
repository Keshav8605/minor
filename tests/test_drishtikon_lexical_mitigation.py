"""
Regression and Precision Tests for DRISHTIKON Single-Token Lexical Collision Mitigation.

Verifies:
1. "Section 144 curfew in Delhi during protests" does NOT retrieve Delhi Metro Phase-IV (Record 1806).
2. Statutory queries (Section 302, Section 377, Section 420) do NOT retrieve Delhi Metro Phase-IV.
3. True positive query "Delhi Metro Phase-IV Janakpuri Krishna Park section" retains Record 1806.
4. "Curfew in Delhi during protests" retains relevant Delhi political protest records.
5. Legitimate cultural queries near the 0.25 threshold (Qutub Minar, Durga Puja, Garba) are retained,
   and "Inner section of Brihadisvara Temple" does not spurious-match Metro Phase-IV.
6. Existing sanitization, legal whitelist, and numeric collision (10,000) behavior is preserved.
7. ContextBuilder PROJECT_JSON fallback functions seamlessly when DRISHTIKON has no match.
"""

import pytest
from pathlib import Path
from src.cultural.context_builder import sanitize_retrieval_query, CulturalContextBuilder
from src.cultural.drishtikon_retriever import DrishtikonRetriever

KB_PATH = "data/cultural/drishtikon/processed/drishtikon_knowledge.json"
CATEGORIES_PATH = "data/cultural/cultural_categories.json"
KNOWLEDGE_PATH = "data/cultural/cultural_knowledge.json"


@pytest.fixture(scope="module")
def retriever():
    r = DrishtikonRetriever(knowledge_path=KB_PATH, min_similarity_threshold=0.25)
    if not r.kb.is_loaded:
        pytest.skip("DRISHTIKON knowledge base not available.")
    return r


@pytest.fixture(scope="module")
def context_builder():
    return CulturalContextBuilder(
        categories_path=CATEGORIES_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        drishtikon_path=KB_PATH,
        drishtikon_enabled=True,
        min_similarity_threshold=0.25
    )


def test_section144_does_not_retrieve_metro_record(retriever):
    """Requirement 1: Section 144 curfew query must not retrieve Delhi Metro Phase-IV (1806)."""
    query = "Section 144 curfew in Delhi during protests"
    san = sanitize_retrieval_query(query)
    matches = retriever.retrieve(san)
    matched_ids = [m["id"] for m in matches]
    assert 1806 not in matched_ids, f"Record 1806 (Delhi Metro) falsely matched for '{query}'"


def test_statutory_sections_do_not_retrieve_metro_record(retriever):
    """Requirement 2: Statutory sections (302, 377, 420) must not retrieve Metro Phase-IV."""
    statutory_queries = [
        "Section 302 IPC murder in Delhi",
        "Section 377 IPC in Delhi",
        "Section 420 IPC fraud in Delhi",
        "Police imposed Section 144 in Mumbai",
        "Section 144 curfew",
    ]
    for q in statutory_queries:
        san = sanitize_retrieval_query(q)
        matches = retriever.retrieve(san)
        matched_ids = [m["id"] for m in matches]
        assert 1806 not in matched_ids, f"Record 1806 falsely matched for statutory query '{q}'"


def test_legitimate_metro_query_retains_metro_record(retriever):
    """Requirement 3: Genuine Delhi Metro Phase-IV query must still retrieve Record 1806."""
    query = "Delhi Metro Phase-IV Janakpuri Krishna Park section"
    san = sanitize_retrieval_query(query)
    matches = retriever.retrieve(san)
    assert len(matches) > 0, "True metro query failed to retrieve any matches"
    top = matches[0]
    assert top["id"] == 1806, f"Expected Record 1806, got ID {top['id']}"
    assert top["relevance_score"] >= 0.45, f"Score {top['relevance_score']} below expected confidence"


def test_curfew_in_delhi_during_protests_retains_protest_results(retriever):
    """Requirement 4: 'Curfew in Delhi during protests' must retain relevant protest results."""
    query = "Curfew in Delhi during protests"
    san = sanitize_retrieval_query(query)
    matches = retriever.retrieve(san)
    assert len(matches) > 0, f"Query '{query}' should retain relevant Delhi protest records"
    top = matches[0]
    # Record 1806 must NOT be matched
    assert top["id"] != 1806
    # Answer should be Political protests
    assert "political protests" in top.get("answer", "").lower()
    assert top["relevance_score"] >= 0.25


def test_legitimate_cultural_queries_near_threshold_retained(retriever):
    """Requirement 5: Cultural queries near 0.25 must be retained; Brihadisvara must not match Metro."""
    # 5a. Qutub Minar
    q_qutub = "Qutub Minar Delhi monument"
    m_qutub = retriever.retrieve(sanitize_retrieval_query(q_qutub))
    assert len(m_qutub) > 0, f"Expected match for '{q_qutub}'"
    assert m_qutub[0]["id"] == 1781

    # 5b. Durga Puja
    q_durga = "Durga Puja in Kolkata West Bengal"
    m_durga = retriever.retrieve(sanitize_retrieval_query(q_durga))
    assert len(m_durga) > 0, f"Expected match for '{q_durga}'"
    assert "west_bengal" in m_durga[0]["state"].lower()

    # 5c. Garba dance
    q_garba = "Garba dance in Gujarat"
    m_garba = retriever.retrieve(sanitize_retrieval_query(q_garba))
    assert len(m_garba) > 0, f"Expected match for '{q_garba}'"
    assert m_garba[0]["id"] == 3249

    # 5d. Inner section of Brihadisvara Temple must NOT match Record 1806
    q_brihad = "Inner section of Brihadisvara Temple"
    m_brihad = retriever.retrieve(sanitize_retrieval_query(q_brihad))
    matched_ids = [m["id"] for m in m_brihad]
    assert 1806 not in matched_ids, f"Brihadisvara temple query falsely matched Delhi Metro (1806)"


def test_existing_sanitization_and_numeric_collision_rules(retriever):
    """Requirement 6: Existing sanitization and numeric collision handling must remain intact."""
    raw = "Protest under Section 144 and Article 370 on 15 August 420 fraud"
    san = sanitize_retrieval_query(raw)
    assert "Section 144" in san
    assert "Article 370" in san
    assert "15 August" in san
    assert "420" in san

    # 10,000 numeric noise stripped and rejected
    noise_raw = "10,000 people gathered in Delhi"
    noise_san = sanitize_retrieval_query(noise_raw)
    assert "10,000" not in noise_san
    m_noise = retriever.retrieve(noise_san)
    assert len(m_noise) == 0, "Numeric noise query should be rejected at 0.25 threshold"


def test_project_json_fallback_functional_when_drishtikon_empty(context_builder):
    """Requirement 7: PROJECT_JSON fallback must remain functional whenever DRISHTIKON has no match."""
    # Section 144 query: DRISHTIKON must NOT inject Metro Phase-IV; PROJECT_JSON handles regional_culture
    prompt_ctx, sources, records = context_builder.build_context("Police imposed Section 144 curfew in Delhi during protests")
    drishtikon_ids = [r.get("id") for r in records]
    assert 1806 not in drishtikon_ids
    assert "DRISHTIKON" not in sources
    assert "PROJECT_JSON" in sources
    assert "PROJECT CULTURAL KNOWLEDGE" in prompt_ctx
    assert "Janakpuri-Krishna Park" not in prompt_ctx

    # Brihadisvara query: DRISHTIKON must NOT inject Metro Phase-IV; PROJECT_JSON handles religion
    prompt_b, sources_b, records_b = context_builder.build_context("Inner section of Brihadisvara Temple")
    assert 1806 not in [r.get("id") for r in records_b]
    assert "DRISHTIKON" not in sources_b
    assert "PROJECT_JSON" in sources_b
    assert "RELIGION" in prompt_b
