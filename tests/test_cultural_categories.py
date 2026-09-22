import unittest
import json
from pathlib import Path

class TestCulturalCategories(unittest.TestCase):
    def test_documents_validity(self):
        cat_path = Path("data/cultural/cultural_categories.json")
        docs_path = Path("data/cultural/cultural_documents.json")

        with open(cat_path, "r", encoding="utf-8") as f:
            categories = json.load(f)

        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        for doc in docs:
            self.assertIn("id", doc)
            self.assertIn("category", doc)
            self.assertIn("title", doc)
            self.assertIn("content", doc)
            self.assertIn(doc["category"], categories, f"'{doc['category']}' is not in the approved taxonomy.")
