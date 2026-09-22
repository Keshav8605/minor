"""
Semantic Context Retriever using ChromaDB and sentence-transformers.
Replaces legacy regex keyword matching with dense vector similarity search.
"""

import logging
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "cultural_knowledge"


class CulturalRetriever:
    def __init__(self, categories_path: str = None, knowledge_path: str = None, chroma_path: str = "data/cultural/chroma_db", distance_threshold: float = 0.65):
        """
        Initializes ChromaDB PersistentClient and embedding function for semantic context retrieval.
        Accepts legacy categories_path and knowledge_path parameters for backward compatibility.
        """
        self.chroma_path = Path(chroma_path)
        self.distance_threshold = distance_threshold

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

    def retrieve_context(self, text: str) -> str:
        """
        Queries ChromaDB for Top-3 semantic matches using cosine similarity.
        Filters out results below the similarity threshold (distance > 0.55).
        Returns formatted bulleted string with header 'EXTERNAL CULTURAL CONTEXT:\n'.
        """
        header = "EXTERNAL CULTURAL CONTEXT:\n"
        fallback = header + "- No specific cultural reference detected."

        if not text or not str(text).strip():
            return fallback

        query_text = str(text).strip()

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=3
            )
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return fallback

        if not results or not results.get("documents") or not results["documents"][0]:
            return fallback

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

        if not matched_bullets:
            return fallback

        return header + "\n".join(matched_bullets)
