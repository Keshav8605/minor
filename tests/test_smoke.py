from src.cultural.category_detector import detect_categories
from src.vlm.output_parser import parse_json_response
import unittest

class TestSmoke(unittest.TestCase):
    def test_smoke_category_detector(self):
        """A minimal smoke test to ensure project components can be instantiated and run."""
        res = detect_categories("नमस्ते")
        self.assertIsInstance(res, list)

    def test_smoke_output_parser(self):
        parsed = parse_json_response('{"humorous": 1, "confidence": 0.9}')
        self.assertEqual(parsed["humorous"], 1)
        self.assertEqual(parsed["confidence"], 0.9)
