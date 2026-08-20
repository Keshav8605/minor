from .knowledge_loader import CulturalKnowledgeBase
from .category_detector import detect_categories

class CulturalRetriever:
    def __init__(self, categories_path: str, knowledge_path: str):
        self.kb = CulturalKnowledgeBase(categories_path, knowledge_path)
        
    def retrieve_context(self, text: str) -> str:
        """
        Detects categories from text and fetches their cultural contexts.
        """
        categories = detect_categories(text)
        
        if "none" in categories and len(categories) == 1:
            return ""
            
        contexts = []
        for cat in categories:
            if cat in self.kb.knowledge and cat != "none":
                contexts.append(f"- {cat.upper()}: {self.kb.knowledge[cat]}")
                
        if not contexts:
            return ""
            
        return "EXTERNAL CULTURAL CONTEXT:\n" + "\n".join(contexts)
