"""
Unit tests for the state manager module.

Tests:
- SHA-256 content-based image identity
- Mode-aware cache store and retrieve
- Deep copy safety (cached results not mutated)
- Cache clearing
"""

import unittest
import os
import tempfile
from app.state_manager import (
    get_image_identity, get_cached_result, store_cached_result,
    clear_cache, MEME_CACHE
)


class TestImageIdentity(unittest.TestCase):
    """Tests for get_image_identity (SHA-256 content hash)."""

    def _create_temp_image(self, content=b"test image data"):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        f.write(content)
        f.close()
        return f.name

    def test_image_identity_deterministic(self):
        """Same file → same hash on repeated calls."""
        path = self._create_temp_image(b"deterministic content")
        try:
            id1 = get_image_identity(path)
            id2 = get_image_identity(path)
            self.assertEqual(id1, id2)
            self.assertEqual(len(id1), 64)  # SHA-256 hex length
        finally:
            os.unlink(path)

    def test_same_content_same_identity(self):
        """Two files with identical content → same identity."""
        path1 = self._create_temp_image(b"same content bytes")
        path2 = self._create_temp_image(b"same content bytes")
        try:
            self.assertEqual(get_image_identity(path1), get_image_identity(path2))
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_different_content_different_identity(self):
        """Two files with different content → different identity."""
        path1 = self._create_temp_image(b"content A unique")
        path2 = self._create_temp_image(b"content B unique")
        try:
            self.assertNotEqual(get_image_identity(path1), get_image_identity(path2))
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestMemeCache(unittest.TestCase):
    """Tests for mode-aware cache operations."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_cache_store_and_retrieve(self):
        """Store and retrieve a complete result dict."""
        result = {
            "prediction": "Humorous",
            "humor_probability": 0.85,
            "non_humor_probability": 0.15,
            "detected_text": "Test meme text",
            "cultural_category": "Family Relations",
            "cultural_dependency": "High",
            "cultural_context": "Indian parenting trope",
            "reasoning": "Humor from relatable pressure."
        }
        store_cached_result("abc123hash", "general", result)
        cached = get_cached_result("abc123hash", "general")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["prediction"], "Humorous")
        self.assertEqual(cached["reasoning"], "Humor from relatable pressure.")
        self.assertAlmostEqual(cached["humor_probability"], 0.85)

    def test_cache_miss_returns_none(self):
        """Unknown identity → None."""
        self.assertIsNone(get_cached_result("nonexistent_hash", "general"))

    def test_general_and_cultural_results_are_separate(self):
        """Same image analyzed in different modes → separate cache entries."""
        general_result = {"prediction": "Not Humorous", "reasoning": "general reasoning"}
        cultural_result = {"prediction": "Humorous", "reasoning": "cultural reasoning with context"}

        store_cached_result("img_hash_1", "general", general_result)
        store_cached_result("img_hash_1", "cultural", cultural_result)

        gen = get_cached_result("img_hash_1", "general")
        cul = get_cached_result("img_hash_1", "cultural")

        self.assertEqual(gen["prediction"], "Not Humorous")
        self.assertEqual(gen["reasoning"], "general reasoning")
        self.assertEqual(cul["prediction"], "Humorous")
        self.assertEqual(cul["reasoning"], "cultural reasoning with context")

    def test_cultural_miss_when_only_general_cached(self):
        """General result cached → cultural mode miss."""
        store_cached_result("img_hash_2", "general", {"prediction": "Humorous"})
        self.assertIsNone(get_cached_result("img_hash_2", "cultural"))

    def test_clear_cache(self):
        """After clear, all lookups return None."""
        store_cached_result("img1", "general", {"prediction": "Humorous"})
        store_cached_result("img2", "cultural", {"prediction": "Not Humorous"})
        self.assertEqual(len(MEME_CACHE), 2)

        clear_cache()

        self.assertEqual(len(MEME_CACHE), 0)
        self.assertIsNone(get_cached_result("img1", "general"))
        self.assertIsNone(get_cached_result("img2", "cultural"))

    def test_cached_result_is_not_mutated(self):
        """Modifying returned copy must NOT affect the canonical cache entry."""
        result = {
            "prediction": "Humorous",
            "reasoning": "original reasoning",
            "detected_text": "original text"
        }
        store_cached_result("img_immutable", "general", result)

        # Get a copy and mutate it
        cached_copy = get_cached_result("img_immutable", "general")
        cached_copy["prediction"] = "MUTATED"
        cached_copy["meme_number"] = 99
        cached_copy["reasoning"] = "tampered reasoning"

        # Original cache must be unchanged
        fresh = get_cached_result("img_immutable", "general")
        self.assertEqual(fresh["prediction"], "Humorous")
        self.assertEqual(fresh["reasoning"], "original reasoning")
        self.assertNotIn("meme_number", fresh)

    def test_store_does_not_retain_reference(self):
        """Mutating the original dict after store must NOT change cache."""
        result = {"prediction": "Humorous", "reasoning": "test"}
        store_cached_result("img_ref", "general", result)

        # Mutate the original dict
        result["prediction"] = "CHANGED"
        result["reasoning"] = "CHANGED"

        # Cache must have the original values
        cached = get_cached_result("img_ref", "general")
        self.assertEqual(cached["prediction"], "Humorous")
        self.assertEqual(cached["reasoning"], "test")


if __name__ == "__main__":
    unittest.main()
