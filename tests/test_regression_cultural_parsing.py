import unittest
import json
from src.vlm.output_parser import parse_json_response, normalize_result_schema
from app.formatting import parse_vlm_output, format_confidence


class TestRegressionCulturalParsing(unittest.TestCase):
    def test_exact_current_bug_regression(self):
        """
        Phase 35: Test exact bug reported in UI.
        Model generates cultural fields + reason.
        Assert that every cultural field is extracted and NO raw JSON leaks into reasoning.
        """
        raw_vlm_response = """
{
  "humorous": true,
  "detected_text": "Sharma ji ka beta got 99% in exams",
  "visual_description": "A stressed Indian teenager being scolded by parents holding report card",
  "cultural_category": "family",
  "cultural_dependency": "high",
  "cultural_context": "Understanding Indian family dynamics and the ubiquitous 'Sharma ji ka beta' comparison trope.",
  "reason": "The humor stems from the relatable pressure in Indian households where academic benchmarks set by peer children create comic domestic tension."
}
"""
        parsed = parse_json_response(raw_vlm_response)

        # 1. Verify structured output parsing
        self.assertNotIn("error", parsed)
        self.assertTrue(parsed["humorous"])
        self.assertEqual(parsed["prediction"], "Humorous")
        self.assertEqual(parsed["detected_text"], "Sharma ji ka beta got 99% in exams")
        self.assertEqual(parsed["cultural_category"], "Family")
        self.assertEqual(parsed["cultural_dependency"], "High")
        self.assertEqual(
            parsed["cultural_context"],
            "Understanding Indian family dynamics and the ubiquitous 'Sharma ji ka beta' comparison trope."
        )
        self.assertEqual(
            parsed["reasoning"],
            "The humor stems from the relatable pressure in Indian households where academic benchmarks set by peer children create comic domestic tension."
        )

        # 2. MUST verify reasoning != entire JSON string and contains no raw braces or JSON markers
        self.assertNotEqual(parsed["reasoning"], raw_vlm_response.strip())
        self.assertFalse(parsed["reasoning"].startswith("{"))
        self.assertNotIn('"humorous": true', parsed["reasoning"])

        # 3. Test UI formatting with 7 fields
        h, c, t, cat, dep, ctx, r = parse_vlm_output(parsed, include_context=True)
        self.assertEqual(h, "Humorous")
        self.assertEqual(t, "Sharma ji ka beta got 99% in exams")
        self.assertEqual(cat, "Family")
        self.assertEqual(dep, "High")
        self.assertEqual(ctx, "Understanding Indian family dynamics and the ubiquitous 'Sharma ji ka beta' comparison trope.")
        self.assertEqual(r, "The humor stems from the relatable pressure in Indian households where academic benchmarks set by peer children create comic domestic tension.")
        self.assertFalse(r.startswith("{"))

    def test_unescaped_newlines_and_control_characters(self):
        """Test model output containing unescaped raw newlines inside strings."""
        raw_text = """{
  "humorous": true,
  "detected_text": "Line 1
Line 2 of Hindi text",
  "cultural_category": "education",
  "cultural_dependency": "medium",
  "cultural_context": "Indian education system
pressure and competition",
  "reason": "Relatable student experience
exam stress humor"
}"""
        parsed = parse_json_response(raw_text)
        self.assertTrue(parsed["humorous"])
        self.assertIn("Line 1", parsed["detected_text"])
        self.assertEqual(parsed["cultural_category"], "Education")
        self.assertEqual(parsed["cultural_dependency"], "Medium")
        self.assertIn("exam stress", parsed["reasoning"])

    def test_markdown_code_block_wrapping(self):
        """Test model wrapping response in ```json ``` markdown code blocks."""
        raw_text = """Here is the analysis of the meme:
```json
{
  "humorous": false,
  "detected_text": "News bulletin headline",
  "visual_description": "News studio",
  "cultural_category": "none",
  "cultural_dependency": "low",
  "cultural_context": "No significant cultural context",
  "reason": "This is a serious informational news report with no comedic intent."
}
```
Hope this is helpful!"""
        parsed = parse_json_response(raw_text)
        self.assertFalse(parsed["humorous"])
        self.assertEqual(parsed["prediction"], "Not Humorous")
        self.assertEqual(parsed["cultural_category"], "No specific cultural category detected")
        self.assertEqual(parsed["cultural_dependency"], "Low")
        self.assertIn("serious informational news", parsed["reasoning"])

    def test_reasoning_key_priority(self):
        """Test that 'reasoning' is prioritized over 'reason' when both appear."""
        data = {
            "humorous": True,
            "reasoning": "Detailed cultural explanation.",
            "reason": "Short explanation."
        }
        norm = normalize_result_schema(data)
        self.assertEqual(norm["reasoning"], "Detailed cultural explanation.")

    def test_cultural_context_used_fallback(self):
        """Test fallback when cultural_context_used contains the context string."""
        data = {
            "humorous": True,
            "cultural_context_used": "IPL cricket rivalry between CSK and RCB",
            "reason": "Cricket rivalry humor"
        }
        norm = normalize_result_schema(data)
        self.assertEqual(norm["cultural_context"], "IPL cricket rivalry between CSK and RCB")

    def test_dependency_normalization(self):
        """Verify low/med/high normalization to Low/Medium/High."""
        for raw, expected in [("low", "Low"), ("LOW", "Low"), ("medium", "Medium"), ("HIGH", "High"), ("none", "Low")]:
            norm = normalize_result_schema({"cultural_dependency": raw})
            self.assertEqual(norm["cultural_dependency"], expected)

    def test_error_handling_no_raw_json_in_ui(self):
        """Verify error result produces clean message, not raw JSON or traceback in UI."""
        error_result = {
            "error": "Failed to parse JSON",
            "raw_response": '{"humorous": true, broken json...'
        }
        h, c, t, cat, dep, ctx, r = parse_vlm_output(error_result, include_context=True)
        self.assertEqual(h, "Error")
        self.assertEqual(c, "N/A")
        self.assertNotIn('broken json', r)
        self.assertNotIn('{', r)
        self.assertEqual(r, "Analysis could not be completed.")

    def test_probability_normalization_display(self):
        """Verify format_confidence computes probabilities summing to ~100%."""
        html = format_confidence(0.769, humor_prob=0.769, non_humor_prob=0.231)
        self.assertIn("76.9%", html)
        self.assertIn("23.1%", html)
        self.assertIn("Model Probability", html)


if __name__ == "__main__":
    unittest.main()
