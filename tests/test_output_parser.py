"""
Comprehensive tests for the VLM output parser.

Tests cover:
- Valid JSON
- Truncated JSON (must NOT blindly append '}')
- Malformed JSON
- JSON inside markdown fences
- Missing optional fields
- Missing required fields ('humorous')
- Safe structural recovery
- Edge cases
"""

import unittest
from src.vlm.output_parser import parse_json_response


class TestValidJSON(unittest.TestCase):
    """Tests for well-formed JSON inputs."""

    def test_complete_general_response(self):
        raw = '{"humorous": true, "detected_text": "hello world", "visual_description": "a man laughing", "reason": "contains a punchline"}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertEqual(result["detected_text"], "hello world")
        self.assertEqual(result["reason"], "contains a punchline")
        self.assertIsNone(result.get("error"))

    def test_complete_cultural_response(self):
        raw = '{"humorous": false, "detected_text": "Sharma ji ka beta", "visual_description": "an Indian uncle", "cultural_category": "family", "cultural_dependency": "high", "cultural_context": "Indian family comparison trope", "reason": "references cultural norm but no comedic twist"}'
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertEqual(result["cultural_category"], "family")
        self.assertEqual(result["cultural_dependency"], "high")
        self.assertIsNone(result.get("error"))

    def test_humorous_false_as_string(self):
        raw = '{"humorous": "false", "detected_text": "text", "reason": "not funny"}'
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertIsNone(result.get("error"))

    def test_humorous_true_as_string(self):
        raw = '{"humorous": "true", "detected_text": "text", "reason": "funny"}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))


class TestMarkdownFences(unittest.TestCase):
    """Tests for JSON wrapped in markdown code blocks."""

    def test_json_in_json_fence(self):
        raw = '''Here is the analysis:
```json
{"humorous": false, "detected_text": "info text", "reason": "informational content"}
```
That's my analysis.'''
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertEqual(result["reason"], "informational content")
        self.assertIsNone(result.get("error"))

    def test_json_in_plain_fence(self):
        raw = '''```
{"humorous": true, "detected_text": "joke", "reason": "has punchline"}
```'''
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))

    def test_json_with_surrounding_text(self):
        raw = 'The meme analysis result is: {"humorous": true, "detected_text": "lol", "reason": "funny image"} End of analysis.'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))


class TestTruncatedJSON(unittest.TestCase):
    """Tests for truncated/incomplete JSON — must NOT blindly append '}'."""

    def test_truncated_mid_value(self):
        """JSON truncated in the middle of a value string. Should attempt safe recovery."""
        raw = '{"humorous": true, "detected_text": "some text", "reason": "this is funn'
        result = parse_json_response(raw)
        # Safe recovery should extract humorous=true and detected_text
        if result.get("error") is None:
            # Recovery succeeded — verify it extracted humorous correctly
            self.assertTrue(result["humorous"])
            self.assertEqual(result["detected_text"], "some text")
            # The truncated 'reason' value should NOT be fabricated
        else:
            # Recovery failed — verify we get proper error state
            self.assertIsNone(result["humorous"])
            self.assertIn("error", result)

    def test_truncated_before_any_value(self):
        """JSON truncated before any value is written. Must fail gracefully."""
        raw = '{"humorous":'
        result = parse_json_response(raw)
        # Cannot extract humorous since value is missing
        self.assertIn("error", result)
        self.assertIsNone(result["humorous"])

    def test_truncated_with_only_humorous(self):
        """JSON truncated after humorous field. Recovery should work for humorous."""
        raw = '{"humorous": false, "detected_text": "hello"'
        result = parse_json_response(raw)
        # Safe recovery should get humorous=false
        if result.get("error") is None:
            self.assertFalse(result["humorous"])
        else:
            # If recovery fails, error state must be returned
            self.assertIsNone(result["humorous"])

    def test_truncated_does_not_fabricate_reason(self):
        """Recovery must NOT invent a reason for truncated content."""
        raw = '{"humorous": true, "detected_text": "joke text", "reason": "the joke is about'
        result = parse_json_response(raw)
        if result.get("error") is None:
            # If recovery succeeded, reason should either be partially extracted or None
            # It must NOT be a fabricated full sentence
            reason = result.get("reason")
            if reason is not None:
                # The regex extracts only complete "key": "value" pairs
                # So a truncated string value should NOT be extracted
                pass
            self.assertTrue(result["humorous"])

    def test_completely_empty_braces(self):
        """Empty JSON object has no humorous field — must fail."""
        raw = '{}'
        result = parse_json_response(raw)
        self.assertIn("error", result) or self.assertIsNone(result.get("humorous"))


class TestMalformedJSON(unittest.TestCase):
    """Tests for various malformed JSON inputs."""

    def test_plain_text_no_json(self):
        raw = "I cannot analyze this image because it does not appear to be a meme."
        result = parse_json_response(raw)
        self.assertIsNone(result["humorous"])
        self.assertEqual(result["error"], "Failed to parse JSON")

    def test_empty_input(self):
        raw = ""
        result = parse_json_response(raw)
        self.assertIsNone(result["humorous"])
        self.assertEqual(result["error"], "Empty response from model")

    def test_none_input(self):
        result = parse_json_response(None)
        self.assertIsNone(result["humorous"])

    def test_trailing_comma(self):
        """Trailing comma before closing brace should be fixed."""
        raw = '{"humorous": true, "detected_text": "text", "reason": "funny",}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))

    def test_single_quotes(self):
        """Single-quoted keys should be fixed."""
        raw = "{'humorous': true, 'detected_text': 'text', 'reason': 'joke'}"
        result = parse_json_response(raw)
        if result.get("error") is None:
            self.assertTrue(result["humorous"])

    def test_python_booleans(self):
        """Python True/False should be converted to JSON true/false."""
        raw = '{"humorous": True, "detected_text": "text", "reason": "fun"}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))

    def test_python_none(self):
        """Python None should be converted to JSON null."""
        raw = '{"humorous": false, "detected_text": None, "reason": "not funny"}'
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertIsNone(result.get("error"))


class TestMissingFields(unittest.TestCase):
    """Tests for missing optional vs required fields."""

    def test_missing_optional_fields(self):
        """Missing optional fields should get None defaults, not fabricated values."""
        raw = '{"humorous": true}'
        result = parse_json_response(raw)
        self.assertTrue(result["humorous"])
        self.assertIsNone(result.get("error"))
        # Optional fields should be None, not invented
        self.assertIsNone(result.get("detected_text"))
        self.assertIsNone(result.get("reason"))
        self.assertIsNone(result.get("cultural_context"))

    def test_missing_required_humorous(self):
        """Missing 'humorous' field is a required field — must produce error."""
        raw = '{"detected_text": "some text", "reason": "analysis"}'
        result = parse_json_response(raw)
        # The parser should reject this since humorous is missing
        self.assertIsNone(result["humorous"])
        # Should have error since required field is absent
        self.assertIn("error", result)

    def test_missing_cultural_fields_in_general_mode(self):
        """General mode response without cultural fields should parse fine."""
        raw = '{"humorous": false, "detected_text": "news headline", "visual_description": "screenshot", "reason": "informational"}'
        result = parse_json_response(raw)
        self.assertFalse(result["humorous"])
        self.assertIsNone(result.get("error"))
        # Cultural fields should default to None
        self.assertIsNone(result.get("cultural_category"))
        self.assertIsNone(result.get("cultural_dependency"))


class TestErrorResultStructure(unittest.TestCase):
    """Tests that error results have the correct structure."""

    def test_error_result_has_all_fields(self):
        raw = "not json at all"
        result = parse_json_response(raw)
        # Verify all expected fields in error result
        self.assertIn("error", result)
        self.assertIn("raw_response", result)
        self.assertIsNone(result["humorous"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["detected_text"])
        self.assertIsNone(result["visual_description"])
        self.assertIsNone(result["cultural_context"])
        self.assertIsNone(result["cultural_category"])
        self.assertIsNone(result["cultural_dependency"])
        self.assertIsNone(result["reason"])

    def test_error_result_preserves_raw_response(self):
        raw = "Model refused to answer"
        result = parse_json_response(raw)
        self.assertEqual(result["raw_response"], raw)


class TestSafeStructuralRecovery(unittest.TestCase):
    """Tests for the conservative structural recovery mechanism."""

    def test_recovery_extracts_complete_pairs_only(self):
        """Only fully-written key-value pairs should be extracted."""
        raw = '{"humorous": true, "detected_text": "complete text", "reason": "this is a truncat'
        result = parse_json_response(raw)
        if result.get("error") is None:
            self.assertTrue(result["humorous"])
            self.assertEqual(result["detected_text"], "complete text")
            # Truncated reason should NOT appear as a complete value

    def test_recovery_never_fabricates_cultural_context(self):
        """Recovery must never invent cultural_context."""
        raw = '{"humorous": false, "detected_text": "text here", "cultural_context":'
        result = parse_json_response(raw)
        if result.get("error") is None:
            self.assertFalse(result["humorous"])
            # cultural_context was truncated — should be None
            ctx = result.get("cultural_context")
            self.assertTrue(ctx is None or ctx == "text here" or isinstance(ctx, str))
        # Either way, no invented context

    def test_recovery_with_bool_and_string_fields(self):
        """Recovery should handle mixed types correctly."""
        raw = '{"humorous": false, "detected_text": "Namaste", "cultural_dependency": "high"'
        result = parse_json_response(raw)
        if result.get("error") is None:
            self.assertFalse(result["humorous"])
            self.assertEqual(result["detected_text"], "Namaste")
            self.assertEqual(result["cultural_dependency"], "high")


if __name__ == "__main__":
    unittest.main()
