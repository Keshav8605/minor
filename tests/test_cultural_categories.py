import unittest
import json
from pathlib import Path
from src.cultural.category_detector import detect_categories

class TestCulturalCategories(unittest.TestCase):
    def test_taxonomy_validity(self):
        cat_path = Path("data/cultural/cultural_categories.json")
        kb_path = Path("data/cultural/cultural_knowledge.json")
        
        with open(cat_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
            
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
            
        for key in kb.keys():
            self.assertIn(key, categories, f"'{key}' is not in the approved taxonomy.")
            
    def test_unicode_word_boundaries(self):
        # English boundaries
        self.assertIn("Bollywood", detect_categories("The famous actor went there."))
        # Devanagari boundaries (was failing with \b)
        self.assertIn("family", detect_categories("वह bhai है।"))
        self.assertIn("education", detect_categories("हमारे teacher कौन हैं?"))
        # Emojis and mixed spacing
        self.assertIn("daily_life", detect_categories("traffic👍"))
        # Substring prevention (should not match if it's inside another word)
        self.assertNotIn("family", detect_categories("moments"))
