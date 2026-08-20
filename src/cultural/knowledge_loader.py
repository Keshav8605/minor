import json
from pathlib import Path

def load_json(filepath: str):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class CulturalKnowledgeBase:
    def __init__(self, categories_path: str, knowledge_path: str):
        self.categories = load_json(categories_path)
        self.knowledge = load_json(knowledge_path)
        
        # Validation
        for key in self.knowledge.keys():
            if key not in self.categories:
                pass # Extraneous key in knowledge base
