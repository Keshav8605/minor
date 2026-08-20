import unittest
from src.vlm.output_parser import parse_json_response

class TestOutputParser(unittest.TestCase):
    def test_valid_json(self):
        raw = '{"humorous": true, "confidence": 0.9, "detected_text": "hello"}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["detected_text"], "hello")

    def test_json_in_markdown(self):
        raw = '''Here is your analysis:
```json
{"humorous": false, "reason": "boring"}
```
Hope this helps!'''
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertEqual(result["reason"], "boring")

    def test_invalid_json_fallback(self):
        raw = "I'm sorry, I cannot parse the image."
        result = parse_json_response(raw)
        self.assertIsNone(result["humorous"])
        self.assertEqual(result["error"], "Failed to parse JSON")
        self.assertEqual(result["raw_response"], raw)
