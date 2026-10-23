"""
Semantic Context Retriever using ChromaDB (when available) and fallback CulturalKnowledgeBase.
Supports dense vector similarity search and deterministic keyword matching.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "cultural_knowledge"

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from .knowledge_loader import CulturalKnowledgeBase
from .category_detector import detect_categories


class CulturalRetriever:
    def __init__(
        self,
        categories_path: str = "data/cultural/cultural_categories.json",
        knowledge_path: str = "data/cultural/cultural_knowledge.json",
        chroma_path: str = "data/cultural/chroma_db",
        distance_threshold: float = 0.65
    ):
        """
        Initializes ChromaDB PersistentClient if available, otherwise falls back
        to keyword-based CulturalKnowledgeBase.
        """
        self.chroma_path = Path(chroma_path)
        self.distance_threshold = distance_threshold
        self.use_chromadb = False
        self.kb = None

        # Load fallback knowledge base if paths exist
        cat_p = categories_path or "data/cultural/cultural_categories.json"
        kno_p = knowledge_path or "data/cultural/cultural_knowledge.json"
        if Path(cat_p).exists() and Path(kno_p).exists():
            try:
                self.kb = CulturalKnowledgeBase(cat_p, kno_p)
            except Exception as e:
                logger.warning("Could not load CulturalKnowledgeBase: %s", e)

        # Try ChromaDB if available and index directory exists
        if _HAS_CHROMADB and self.chroma_path.exists():
            try:
                logger.info("Initializing CulturalRetriever with ChromaDB at %s...", self.chroma_path)
                self.client = chromadb.PersistentClient(path=str(self.chroma_path.resolve()))
                self.embedding_func = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
                try:
                    self.collection = self.client.get_collection(
                        name=COLLECTION_NAME,
                        embedding_function=self.embedding_func
                    )
                except Exception:
                    self.collection = self.client.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=self.embedding_func,
                        metadata={"hnsw:space": "cosine"}
                    )
                self.use_chromadb = True
            except Exception as e:
                logger.warning("ChromaDB initialization failed, falling back to keyword matcher: %s", e)
                self.use_chromadb = False

    def retrieve_context(self, text: str) -> str:
        """
        Retrieves cultural context for text.
        Returns formatted bulleted string with header 'EXTERNAL CULTURAL CONTEXT:\\n'.
        """
        header = "EXTERNAL CULTURAL CONTEXT:\n"
        fallback = header + "- No specific cultural reference detected."

        if not text or not str(text).strip():
            return fallback

        query_text = str(text).strip()

        # ChromaDB path
        if self.use_chromadb:
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=3
                )
                if results and results.get("documents") and results["documents"][0]:
                    matched_bullets = []
                    docs = results["documents"][0]
                    distances = results.get("distances", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]

                    for i, doc_text in enumerate(docs):
                        dist = distances[i] if i < len(distances) else 0.0
                        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

                        if dist <= self.distance_threshold:
                            cat = meta.get("category", "")
                            if cat:
                                formatted_line = f"- {cat.upper()}: {doc_text}"
                            else:
                                formatted_line = f"- {doc_text}"
                            matched_bullets.append(formatted_line)

                    if matched_bullets:
                        return header + "\n".join(matched_bullets)
            except Exception as e:
                logger.warning("ChromaDB query failed: %s", e)

        # Fallback keyword matching
        if self.kb is not None:
            categories = detect_categories(query_text)
            if "none" in categories and len(categories) == 1:
                return fallback

            contexts = []
            for cat in categories:
                if cat in self.kb.knowledge and cat != "none":
                    contexts.append(f"- {cat.upper()}: {self.kb.knowledge[cat]}")

            if contexts:
                return header + "\n".join(contexts)

        return fallback
