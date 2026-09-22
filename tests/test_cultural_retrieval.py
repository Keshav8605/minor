import unittest
from pathlib import Path
from src.cultural.context_retriever import CulturalRetriever

class TestCulturalRetrieval(unittest.TestCase):
    def setUp(self):
        chroma_path = "data/cultural/chroma_db"
        if Path(chroma_path).exists():
            self.retriever = CulturalRetriever(chroma_path=chroma_path)
        else:
            self.retriever = None

    def test_semantic_retrieval(self):
        if not self.retriever:
            self.skipTest("ChromaDB index missing")

        context = self.retriever.retrieve_context("Dhoni hit a helicopter shot in the IPL.")
        self.assertTrue(context.startswith("EXTERNAL CULTURAL CONTEXT:\n"))
        self.assertIn("CRICKET", context)

        # Test threshold fallback for unrelated text
        context_unrelated = self.retriever.retrieve_context("xyzabc12345 non-existent random string")
        self.assertIn("No specific cultural reference detected", context_unrelated)
