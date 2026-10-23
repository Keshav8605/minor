"""
Unit tests for retrieval query sanitization in CulturalContextBuilder.
"""

import pytest
from src.cultural.context_builder import sanitize_retrieval_query


def test_strip_generic_meme_prefixes():
    # "Me and the boys after..."
    raw = "Me and the boys after earning 10,000 for reporting fake Whatsapp videos"
    clean = sanitize_retrieval_query(raw)
    assert "me and the boys" not in clean.lower()
    assert "whatsapp videos" in clean.lower()

    # "Nobody:"
    raw = "Nobody: Me during quarantine: eating all snacks"
    clean = sanitize_retrieval_query(raw)
    assert "nobody:" not in clean.lower()
    assert "quarantine" in clean.lower()

    # "Le husband:"
    raw = "Le husband: main baad me phone karta hoon"
    clean = sanitize_retrieval_query(raw)
    assert "le husband:" not in clean.lower()
    assert "phone karta hoon" in clean.lower()

    # "POV:"
    raw = "POV: You forgot to study for JEE exam"
    clean = sanitize_retrieval_query(raw)
    assert "pov:" not in clean.lower()
    assert "jee exam" in clean.lower()


def test_filter_isolated_numeric_noise():
    # Standalone large numbers that cause false lexical collisions
    raw = "Earned 10,000 cash prize with 50000 views on 21 days"
    clean = sanitize_retrieval_query(raw)
    assert "10,000" not in clean
    assert "50000" not in clean
    # 21 was isolated
    assert " 21 " not in clean
    assert "earned" in clean.lower()
    assert "cash prize" in clean.lower()


def test_preserve_legal_sections_and_articles():
    # Legal sections and articles must be kept intact
    t1 = sanitize_retrieval_query("Police imposed Section 144 in Delhi")
    assert "Section 144" in t1

    t2 = sanitize_retrieval_query("Debate on Article 370 in Kashmir")
    assert "Article 370" in t2

    t3 = sanitize_retrieval_query("Filed under Dhara 420 for fraud")
    assert "Dhara 420" in t3 or "420" in t3


def test_preserve_cultural_numbers():
    # 420 (fraud/trickery), 370 (special status), 144 (curfew)
    assert "420" in sanitize_retrieval_query("Bhai yeh toh 420 aadmi hai")
    assert "370" in sanitize_retrieval_query("Removal of 370")
    assert "144" in sanitize_retrieval_query("Curfew under 144")


def test_preserve_calendar_dates():
    # Calendar dates with months
    t1 = sanitize_retrieval_query("Independence Day on 15 August 2026")
    assert "15 August" in t1

    t2 = sanitize_retrieval_query("Celebrations on 26th January Republic Day")
    assert "26th January" in t2


def test_preserve_hinglish_slang_and_names():
    raw = "PM Modi Speech Live: Nibbi Nibba BABY HUM OR 21 DIN NAHI MIL SAKTE"
    clean = sanitize_retrieval_query(raw)
    assert "PM Modi" in clean
    assert "Nibbi" in clean
    assert "Nibba" in clean
    assert "BABY HUM OR" in clean
    assert "DIN NAHI MIL SAKTE" in clean


def test_empty_and_whitespace_inputs():
    assert sanitize_retrieval_query("") == ""
    assert sanitize_retrieval_query("   ") == ""
    assert sanitize_retrieval_query(None) == ""
