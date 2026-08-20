import unittest
from app.formatting import format_confidence, parse_vlm_output

class TestUIFormatting(unittest.TestCase):
    def test_format_confidence(self):
        self.assertIn("#10b981", format_confidence(0.9))
        self.assertIn("#f59e0b", format_confidence(0.6))
        self.assertIn("#ef4444", format_confidence(0.2))
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
        self.assertIn("#10b981", c)
        self.assertEqual(t, "Hello")
        self.assertEqual(cat, "Category: Family | Relevance: High")
        self.assertEqual(dep, "")
        self.assertEqual(r, "funny")
        
    def test_parse_error_output(self):
        mock_result = {"error": True, "raw_response": "Timeout"}
        h, c, t, cat, dep, r = parse_vlm_output(mock_result)
        self.assertEqual(h, "Error")
        self.assertEqual(r, "Timeout")
        
    def test_parse_invalid_format(self):
        h, c, t, cat, dep, r = parse_vlm_output("Not a dict")
        self.assertEqual(h, "Error")
