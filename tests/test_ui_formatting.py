import unittest
import os
import tempfile
from PIL import Image
from app.formatting import format_confidence, parse_vlm_output
from app.ui_components import _format_file_size, _get_image_data_uri, _render_multi_meme_cards_html


class TestUIFormatting(unittest.TestCase):
    def test_format_confidence(self):
        self.assertIn("#22C55E", format_confidence(0.9))
        self.assertIn("#F59E0B", format_confidence(0.6))
        self.assertIn("#EF4444", format_confidence(0.2))
        self.assertIn("N/A", format_confidence("N/A"))
        
    def test_parse_valid_output(self):
        mock_result = {
            "humorous": True,
            "confidence": 0.85,
            "detected_text": "Hello",
            "cultural_category": "family",
            "cultural_dependency": "high",
            "reason": "funny"
        }
        
        h, c, t, cat, dep, r = parse_vlm_output(mock_result)
        self.assertEqual(h, "Humorous")
        self.assertIn("#22C55E", c)
        self.assertEqual(t, "Hello")
        self.assertEqual(cat, "Family")
        self.assertEqual(dep, "High")
        self.assertEqual(r, "funny")
        
    def test_parse_error_output(self):
        mock_result = {"error": True, "raw_response": "Timeout"}
        h, c, t, cat, dep, r = parse_vlm_output(mock_result)
        self.assertEqual(h, "Error")
        self.assertEqual(r, "Timeout")
        
    def test_parse_invalid_format(self):
        h, c, t, cat, dep, r = parse_vlm_output("Not a dict")
        self.assertEqual(h, "Error")

    def test_format_file_size(self):
        self.assertEqual(_format_file_size(500), "500 B")
        self.assertEqual(_format_file_size(1024 * 50), "50.0 KB")
        self.assertEqual(_format_file_size(1024 * 1024 * 2.5), "2.5 MB")
        self.assertEqual(_format_file_size(0), "")

    def test_render_multi_meme_cards_html(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "meme_a.png")
            p2 = os.path.join(td, "meme_b.jpg")
            
            img1 = Image.new("RGB", (100, 100), color="blue")
            img1.save(p1)
            img2 = Image.new("RGB", (120, 80), color="red")
            img2.save(p2)
            
            html = _render_multi_meme_cards_html([p1, p2])
            
            self.assertIn("Uploaded Memes", html)
            self.assertIn("Preserved Input Order", html)
            self.assertIn("2 memes", html)
            self.assertIn("MEME 1", html)
            self.assertIn("MEME 2", html)
            self.assertIn("meme_a.png", html)
            self.assertIn("meme_b.jpg", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("data:image/jpeg;base64,", html)
            self.assertIn("removeMultiMemeFile", html)
            self.assertIn("KB", html)

    def test_render_multi_meme_cards_empty(self):
        self.assertEqual(_render_multi_meme_cards_html([]), "")
        self.assertEqual(_render_multi_meme_cards_html(None), "")

