from src.cultural.context_retriever import CulturalRetriever
from src.vlm.output_parser import parse_json_response
import unittest

class TestSmoke(unittest.TestCase):
    def test_smoke_cultural_retriever(self):
        """A minimal smoke test to ensure CulturalRetriever can be instantiated and queried."""
        retriever = CulturalRetriever()
        res = retriever.retrieve_context("नमस्ते")
        self.assertTrue(res.startswith("EXTERNAL CULTURAL CONTEXT:\n"))

    def test_smoke_output_parser(self):
        parsed = parse_json_response('{"humorous": 1, "confidence": 0.9}')
        self.assertEqual(parsed["humorous"], 1)
        self.assertEqual(parsed["confidence"], 0.9)
