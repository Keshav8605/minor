"""
Regression tests for Multi-Image / Multi-Meme Separation.

Verifies:
1. Exact input order preservation (Meme 1 -> img 1, Meme 2 -> img 2, Meme 3 -> img 3).
2. Independent per-image processing: zero cross-meme data mixing in OCR, cultural context, and reasoning.
3. Structured multi-meme schema: 'memes' list, no redundant 'confidence' field, presence of humor_probability and non_humor_probability.
4. Single-image backward compatibility (single-image behavior unchanged).
5. HTML multi-meme card formatting with prominent numbered headers and clear dividers.
6. SHA-256 content-hash caching: cache hits reuse results, cache misses trigger analysis.
7. Mode-aware caching: General and Cultural-Aware results are separate.
8. Single/Multi state separation: tab state determines dispatch, no stale state leakage.
9. Reasoning preservation: every meme has its own reasoning.
"""

import unittest
from unittest.mock import MagicMock, patch
from app.app import AnalysisResponse, analyze_meme, _build_meme_record
from app.formatting import format_multi_meme_cards, parse_vlm_output
from app.state_manager import clear_cache


class TestMultiMemeSeparation(unittest.TestCase):

    def setUp(self):
        """Clear cache and patch get_image_identity for tests using mock file paths."""
        clear_cache()
        # Patch get_image_identity to use path-based identity (no file I/O needed)
        self._identity_patcher = patch(
            "app.app.get_image_identity",
            side_effect=lambda p: f"mock_hash_{str(p)}"
        )
        self._identity_patcher.start()

    def tearDown(self):
        self._identity_patcher.stop()
        clear_cache()

    def test_single_image_backward_compatibility(self):
        """Verify that when 1 image is provided, single-meme behavior is preserved."""
        mock_single_res = {
            "humorous": True,
            "prediction": "Humorous",
            "confidence": 0.88,
            "humor_probability": 0.88,
            "non_humor_probability": 0.12,
            "detected_text": "Single meme text",
            "cultural_category": "Family Relations",
            "cultural_dependency": "High",
            "cultural_context": "Indian parenting comparison",
            "reasoning": "Humor comes from relatable family pressure."
        }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("Single meme text", "EasyOCR")), \
             patch("os.path.exists", return_value=True):
            
            mock_vlm.infer.return_value = mock_single_res

            resp = analyze_meme("dummy/path/meme1.jpg", cultural_mode="Cultural-Aware")

            # 1. Backward-compatible 7-tuple unpacking
            h, c, t, cat, dep, ctx, r = resp
            self.assertEqual(h, "Humorous")
            self.assertEqual(t, "Single meme text")
            self.assertEqual(cat, "Family Relations")
            self.assertEqual(dep, "High")
            self.assertEqual(ctx, "Indian parenting comparison")
            self.assertEqual(r, "Humor comes from relatable family pressure.")

            # 2. Dictionary properties
            self.assertFalse(resp.get("is_multi"))
            self.assertEqual(resp.get("memes_count"), 1)
            self.assertEqual(len(resp.get("memes")), 1)
            self.assertEqual(resp.get("memes")[0]["meme_number"], 1)

    def test_two_images_order_and_data_isolation(self):
        """
        Verify that 2 images (Family and Cricket) preserve exact order
        and that OCR, cultural context, and reasoning are strictly isolated.
        """
        family_res = {
            "humorous": True,
            "prediction": "Humorous",
            "humor_probability": 0.85,
            "non_humor_probability": 0.15,
            "detected_text": "Sharma ji ka beta got 99%",
            "cultural_category": "Family Relations",
            "cultural_dependency": "High",
            "cultural_context": "Indian parental comparison trope",
            "reasoning": "Relatable Indian parenting academic pressure."
        }
        cricket_res = {
            "humorous": False,
            "prediction": "Not Humorous",
            "humor_probability": 0.10,
            "non_humor_probability": 0.90,
            "detected_text": "Rahul Dravid voting line",
            "cultural_category": "Cricket Sports",
            "cultural_dependency": "Medium",
            "cultural_context": "Indian cricket icon civic duty",
            "reasoning": "Factual news report regarding election icon."
        }

        def mock_infer_side_effect(path, mode="general", ocr_text="", retrieved_context=""):
            if "family" in str(path):
                return dict(family_res)
            return dict(cricket_res)

        def mock_ocr_side_effect(path):
            if "family" in str(path):
                return "Sharma ji ka beta got 99%", "EasyOCR"
            return "Rahul Dravid voting line", "EasyOCR"

        mock_retriever = MagicMock()
        mock_retriever.retrieve_context.side_effect = lambda txt: "Indian parental comparison trope" if "Sharma" in txt else "Indian cricket icon civic duty"

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("app.app.CULTURAL_RETRIEVER", mock_retriever), \
             patch("src.cultural.ocr_engine.extract_text_ocr", side_effect=mock_ocr_side_effect), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer_side_effect

            image_paths = ["data/family_meme.jpg", "data/cricket_meme.jpg"]
            resp = analyze_meme(image_paths, cultural_mode="Cultural-Aware")

            self.assertTrue(resp.get("is_multi"))
            self.assertEqual(resp.get("memes_count"), 2)
            memes = resp.get("memes")
            self.assertEqual(len(memes), 2)

            # Check Meme 1: ONLY family data
            meme_1 = memes[0]
            self.assertEqual(meme_1["meme_number"], 1)
            self.assertIn("family", meme_1["image_path"])
            self.assertEqual(meme_1["detected_text"], "Sharma ji ka beta got 99%")
            self.assertEqual(meme_1["cultural_category"], "Family Relations")
            self.assertEqual(meme_1["cultural_context"], "Indian parental comparison trope")
            self.assertEqual(meme_1["reasoning"], "Relatable Indian parenting academic pressure.")
            self.assertNotIn("Rahul Dravid", meme_1["detected_text"])
            self.assertNotIn("cricket", meme_1["cultural_context"].lower())

            # Check Meme 2: ONLY cricket data
            meme_2 = memes[1]
            self.assertEqual(meme_2["meme_number"], 2)
            self.assertIn("cricket", meme_2["image_path"])
            self.assertEqual(meme_2["detected_text"], "Rahul Dravid voting line")
            self.assertEqual(meme_2["cultural_category"], "Cricket Sports")
            self.assertEqual(meme_2["cultural_context"], "Indian cricket icon civic duty")
            self.assertEqual(meme_2["reasoning"], "Factual news report regarding election icon.")
            self.assertNotIn("Sharma ji", meme_2["detected_text"])
            self.assertNotIn("parent", meme_2["cultural_context"].lower())

            # Verify no redundant 'confidence' field in meme schema
            self.assertNotIn("confidence", meme_1)
            self.assertNotIn("confidence", meme_2)
            self.assertIn("humor_probability", meme_1)
            self.assertIn("non_humor_probability", meme_1)

    def test_three_images_order_and_numbering(self):
        """Verify 3 images are numbered 1, 2, 3 in exact order."""
        paths = ["img_a_family.jpg", "img_b_kota.jpg", "img_c_cricket.jpg"]

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("Sample text", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.return_value = {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.75, "non_humor_probability": 0.25,
                "detected_text": "Sample text", "cultural_category": "Other",
                "cultural_dependency": "Low", "cultural_context": "Sample context",
                "reasoning": "Sample reasoning"
            }

            resp = analyze_meme(paths, cultural_mode="General")
            self.assertTrue(resp.get("is_multi"))
            self.assertEqual(resp.get("memes_count"), 3)
            memes = resp.get("memes")
            self.assertEqual([m["meme_number"] for m in memes], [1, 2, 3])
            self.assertEqual(memes[0]["image_path"], "img_a_family.jpg")
            self.assertEqual(memes[1]["image_path"], "img_b_kota.jpg")
            self.assertEqual(memes[2]["image_path"], "img_c_cricket.jpg")

    def test_html_multi_meme_card_formatting(self):
        """Verify format_multi_meme_cards produces separate numbered sections with headers and dividers."""
        multi_result = {
            "is_multi": True,
            "memes": [
                {
                    "meme_number": 1,
                    "image_path": "uploads/family.jpg",
                    "prediction": "Humorous",
                    "humor_probability": 0.82,
                    "non_humor_probability": 0.18,
                    "detected_text": "Beta 95% kyu aaye",
                    "cultural_category": "Family Relations",
                    "cultural_dependency": "High",
                    "cultural_context": "Desi parenting academic benchmarks",
                    "reasoning": "Depicts comic tension over marks."
                },
                {
                    "meme_number": 2,
                    "image_path": "uploads/cricket.jpg",
                    "prediction": "Not Humorous",
                    "humor_probability": 0.15,
                    "non_humor_probability": 0.85,
                    "detected_text": "Cricket match score update",
                    "cultural_category": "Cricket Sports",
                    "cultural_dependency": "Low",
                    "cultural_context": "Indian cricket score update",
                    "reasoning": "Informational text about sports score."
                }
            ]
        }

        html = format_multi_meme_cards(multi_result, include_context=True)

        # 1. Distinct numbered section headers
        self.assertIn("MEME 1 ANALYSIS", html)
        self.assertIn("MEME 2 ANALYSIS", html)

        # 2. Image filenames / badges
        self.assertIn("family.jpg", html)
        self.assertIn("cricket.jpg", html)

        # 3. Card labels present in each section
        self.assertIn("🎯 HUMOR PREDICTION", html)
        self.assertIn("◉ MODEL PROBABILITY", html)
        self.assertIn("📄 DETECTED TEXT (OCR)", html)
        self.assertIn("🏷️ CULTURAL CATEGORY", html)
        self.assertIn("🔗 CULTURAL DEPENDENCY", html)
        self.assertIn("🌐 CULTURAL CONTEXT", html)
        self.assertIn("🧠 AI REASONING", html)

        # 4. Probabilities rendered
        self.assertIn("82.0%", html)
        self.assertIn("18.0%", html)
        self.assertIn("15.0%", html)
        self.assertIn("85.0%", html)

        # 5. Clear separator between memes
        self.assertIn("meme-separator", html)


class TestIncrementalCaching(unittest.TestCase):
    """Tests for cache-based incremental multi-image analysis."""

    def setUp(self):
        clear_cache()
        self._identity_patcher = patch(
            "app.app.get_image_identity",
            side_effect=lambda p: f"mock_hash_{str(p)}"
        )
        self._identity_patcher.start()

    def tearDown(self):
        self._identity_patcher.stop()
        clear_cache()

    def test_cache_hit_reuses_result(self):
        """Adding a 3rd image should only trigger VLM inference for the new image."""
        call_count = {"value": 0}

        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            call_count["value"] += 1
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": f"Text from {path}",
                "cultural_category": "Other", "cultural_dependency": "Low",
                "cultural_context": "Context", "reasoning": f"Reasoning for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("Text", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # First call: analyze A + B → 2 VLM calls
            resp1 = analyze_meme(["a.jpg", "b.jpg"], "General")
            first_calls = call_count["value"]
            self.assertEqual(first_calls, 2)
            self.assertEqual(resp1.get("memes_count"), 2)

            # Second call: analyze A + B + C → only 1 new VLM call (C)
            resp2 = analyze_meme(["a.jpg", "b.jpg", "c.jpg"], "General")
            second_new_calls = call_count["value"] - first_calls
            self.assertEqual(second_new_calls, 1)  # Only C was analyzed
            self.assertEqual(resp2.get("memes_count"), 3)

    def test_cache_preserves_order(self):
        """Cached results maintain the current upload order numbering."""
        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.9, "non_humor_probability": 0.1,
                "detected_text": f"Text_{path}", "cultural_category": "Other",
                "cultural_dependency": "Low", "cultural_context": "Ctx",
                "reasoning": f"Reason for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # Analyze A + B
            analyze_meme(["a.jpg", "b.jpg"], "General")

            # Now analyze A + B + C (A, B cached; C new)
            resp = analyze_meme(["a.jpg", "b.jpg", "c.jpg"], "General")
            memes = resp.get("memes")

            self.assertEqual(memes[0]["meme_number"], 1)
            self.assertEqual(memes[0]["image_path"], "a.jpg")
            self.assertEqual(memes[1]["meme_number"], 2)
            self.assertEqual(memes[1]["image_path"], "b.jpg")
            self.assertEqual(memes[2]["meme_number"], 3)
            self.assertEqual(memes[2]["image_path"], "c.jpg")

    def test_image_removal_updates_numbering(self):
        """Removing an image from the middle renumbers correctly."""
        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.7, "non_humor_probability": 0.3,
                "detected_text": f"Text_{path}", "cultural_category": "Other",
                "cultural_dependency": "Low", "cultural_context": "Ctx",
                "reasoning": f"Reason for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # Analyze A + B + C
            analyze_meme(["a.jpg", "b.jpg", "c.jpg"], "General")

            # Remove B → A + C
            resp = analyze_meme(["a.jpg", "c.jpg"], "General")
            memes = resp.get("memes")

            self.assertEqual(len(memes), 2)
            self.assertEqual(memes[0]["meme_number"], 1)
            self.assertEqual(memes[0]["image_path"], "a.jpg")
            self.assertEqual(memes[1]["meme_number"], 2)
            self.assertEqual(memes[1]["image_path"], "c.jpg")

    def test_cache_miss_for_new_image(self):
        """A brand new image triggers full analysis even if others are cached."""
        call_count = {"value": 0}

        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            call_count["value"] += 1
            return {
                "humorous": False, "prediction": "Not Humorous",
                "humor_probability": 0.3, "non_humor_probability": 0.7,
                "detected_text": f"Text_{path}", "cultural_category": "Other",
                "cultural_dependency": "Low", "cultural_context": "Ctx",
                "reasoning": f"Reason for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # Analyze A
            analyze_meme("a.jpg", "General")
            self.assertEqual(call_count["value"], 1)

            # Analyze completely new B
            analyze_meme("b.jpg", "General")
            self.assertEqual(call_count["value"], 2)

    def test_mode_aware_cache_separation(self):
        """General result for an image must NOT be reused for Cultural-Aware analysis."""
        call_count = {"value": 0}

        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            call_count["value"] += 1
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": "Text", "cultural_category": "Other",
                "cultural_dependency": "Low",
                "cultural_context": f"Context in {mode} mode",
                "reasoning": f"Reasoning in {mode} mode"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("Text", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # Analyze in General mode
            analyze_meme("same_image.jpg", "General")
            self.assertEqual(call_count["value"], 1)

            # Analyze SAME image in Cultural-Aware mode → must NOT use General cache
            analyze_meme("same_image.jpg", "Cultural-Aware")
            self.assertEqual(call_count["value"], 2)

            # Analyze again in General mode → SHOULD use General cache
            analyze_meme("same_image.jpg", "General")
            self.assertEqual(call_count["value"], 2)  # No new call

    def test_single_image_caching(self):
        """Single-image results should also be cached and reusable."""
        call_count = {"value": 0}

        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            call_count["value"] += 1
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.9, "non_humor_probability": 0.1,
                "detected_text": "Cached text", "cultural_category": "Other",
                "cultural_dependency": "Low", "cultural_context": "Cached context",
                "reasoning": "Cached reasoning"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            # First analysis
            resp1 = analyze_meme("single.jpg", "General")
            self.assertEqual(call_count["value"], 1)

            # Second analysis of same image — cached
            resp2 = analyze_meme("single.jpg", "General")
            self.assertEqual(call_count["value"], 1)  # No new VLM call

            # Results should be identical
            self.assertEqual(resp1[0], resp2[0])


class TestStateSeparation(unittest.TestCase):
    """Tests for single/multi state separation and reasoning."""

    def setUp(self):
        clear_cache()
        self._identity_patcher = patch(
            "app.app.get_image_identity",
            side_effect=lambda p: f"mock_hash_{str(p)}"
        )
        self._identity_patcher.start()

    def tearDown(self):
        self._identity_patcher.stop()
        clear_cache()

    def test_single_mode_no_multi_state(self):
        """Single-image analysis must return is_multi=False."""
        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.return_value = {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": "T", "reasoning": "R"
            }

            resp = analyze_meme("single.jpg", "General")
            self.assertFalse(resp.get("is_multi"))
            self.assertEqual(resp.get("memes_count"), 1)

    def test_reasoning_always_present(self):
        """Every meme in multi-result must have non-empty reasoning from the VLM."""
        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": "Some text",
                "reasoning": f"Specific reasoning for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            resp = analyze_meme(["m1.jpg", "m2.jpg", "m3.jpg"], "General")
            memes = resp.get("memes")

            for i, meme in enumerate(memes):
                self.assertIn("reasoning", meme)
                self.assertTrue(len(meme["reasoning"]) > 0,
                                f"Meme {i+1} has empty reasoning")
                self.assertNotEqual(meme["reasoning"], "No reasoning provided.",
                                    f"Meme {i+1} reasoning was lost")

    def test_reasoning_is_meme_specific(self):
        """Each meme's reasoning must be unique and correspond only to that meme."""
        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": f"Text_{path}",
                "reasoning": f"Unique reasoning specifically for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            resp = analyze_meme(["alpha.jpg", "beta.jpg", "gamma.jpg"], "General")
            memes = resp.get("memes")

            # Each meme has its own unique reasoning
            reasons = [m["reasoning"] for m in memes]
            self.assertEqual(len(set(reasons)), 3, "Reasoning should be unique per meme")
            self.assertIn("alpha.jpg", reasons[0])
            self.assertIn("beta.jpg", reasons[1])
            self.assertIn("gamma.jpg", reasons[2])

    def test_result_schema_no_confidence(self):
        """Multi-meme schema must NOT contain a 'confidence' field."""
        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.return_value = {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": "T", "reasoning": "R"
            }

            resp = analyze_meme(["a.jpg", "b.jpg"], "General")
            for meme in resp.get("memes"):
                self.assertNotIn("confidence", meme)
                self.assertIn("humor_probability", meme)
                self.assertIn("non_humor_probability", meme)

    def test_multi_result_count_matches_current_images(self):
        """The memes count must match the CURRENT number of uploaded images."""
        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.return_value = {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": "T", "reasoning": "R"
            }

            # 3 images → 3 results
            resp3 = analyze_meme(["a.jpg", "b.jpg", "c.jpg"], "General")
            self.assertEqual(resp3.get("memes_count"), 3)

            # Then only 2 images → 2 results (not 3)
            resp2 = analyze_meme(["a.jpg", "c.jpg"], "General")
            self.assertEqual(resp2.get("memes_count"), 2)

    def test_meme_number_mapping(self):
        """Upload area numbering must match result card numbering."""
        def mock_infer(path, mode="general", ocr_text="", retrieved_context=""):
            return {
                "humorous": True, "prediction": "Humorous",
                "humor_probability": 0.8, "non_humor_probability": 0.2,
                "detected_text": f"Text from {path}", "reasoning": f"Reason for {path}"
            }

        with patch("app.app.init_engine", return_value=True), \
             patch("app.app.VLM_ENGINE") as mock_vlm, \
             patch("src.cultural.ocr_engine.extract_text_ocr", return_value=("T", "EasyOCR")), \
             patch("os.path.exists", return_value=True):

            mock_vlm.infer.side_effect = mock_infer

            paths = ["imageA.jpg", "imageB.jpg", "imageC.jpg"]
            resp = analyze_meme(paths, "General")
            memes = resp.get("memes")

            # Verify 1-to-1 mapping: meme_number → image_path
            for idx, (path, meme) in enumerate(zip(paths, memes)):
                self.assertEqual(meme["meme_number"], idx + 1)
                self.assertEqual(meme["image_path"], path)


class TestBuildMemeRecord(unittest.TestCase):
    """Tests for the _build_meme_record helper function."""

    def test_strips_confidence_field(self):
        """_build_meme_record must not include 'confidence' in its output."""
        vlm_result = {
            "humorous": True, "prediction": "Humorous",
            "confidence": 0.92,
            "humor_probability": 0.92, "non_humor_probability": 0.08,
            "detected_text": "Test", "cultural_category": "Other",
            "cultural_dependency": "Low", "cultural_context": "None",
            "reasoning": "Test reasoning"
        }
        record = _build_meme_record(vlm_result, idx=1, path="test.jpg")
        self.assertNotIn("confidence", record)
        self.assertEqual(record["meme_number"], 1)
        self.assertEqual(record["image_path"], "test.jpg")

    def test_preserves_all_required_fields(self):
        """All required schema fields must be present in the record."""
        vlm_result = {
            "humorous": True, "prediction": "Humorous",
            "humor_probability": 0.8, "non_humor_probability": 0.2,
            "detected_text": "OCR text", "cultural_category": "Family",
            "cultural_dependency": "High", "cultural_context": "Context",
            "reasoning": "Full reasoning text", "timing": {"inference_s": 2.5}
        }
        record = _build_meme_record(vlm_result, idx=3, path="img3.jpg")

        required = [
            "meme_number", "image_path", "humorous", "prediction",
            "humor_probability", "non_humor_probability", "detected_text",
            "cultural_category", "cultural_dependency", "cultural_context",
            "reasoning", "timing"
        ]
        for field in required:
            self.assertIn(field, record, f"Missing field: {field}")


if __name__ == "__main__":
    unittest.main()
