import unittest
from src.cultural.category_detector import detect_categories
from src.cultural.context_retriever import CulturalRetriever
from pathlib import Path

class TestCulturalRetrieval(unittest.TestCase):
    def setUp(self):
        cat_path = "data/cultural/cultural_categories.json"
        kb_path = "data/cultural/cultural_knowledge.json"
        
        if Path(cat_path).exists() and Path(kb_path).exists():
            self.retriever = CulturalRetriever(cat_path, kb_path)
        else:
            self.retriever = None
            
    def test_detect_category(self):
        cats = detect_categories("I hate going to the office, my boss is mean.")
        self.assertIn("workplace", cats)
        
        cats = detect_categories("Sharma ji ka beta got 99 percent in JEE.")
        self.assertIn("family", cats)
        self.assertIn("JEE/exams", cats)
        
        cats = detect_categories("Just eating an apple.")
        self.assertIn("none", cats)
        
    def test_retrieval_string(self):
        if not self.retriever:
            self.skipTest("Knowledge files missing")
            
        context = self.retriever.retrieve_context("Dhoni hit a six in the IPL.")
        self.assertIn("CRICKET", context)
        self.assertIn("Dhoni", context)
