"""
DRISHTIKON Cultural Context Retriever.

Uses TF-IDF vectorization and cosine similarity over deterministically constructed
'search_text' to retrieve the most relevant DRISHTIKON cultural records.

Preserves full source traceability (source, id, state, attribute, language,
question, answer, image_name, image_link, relevance_score).

Does NOT fabricate explanations or invent facts.
"""

import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .drishtikon_loader import DrishtikonKnowledgeBase

logger = logging.getLogger(__name__)


class DrishtikonRetriever:
    """
    Lightweight, deterministic TF-IDF retriever for DRISHTIKON records.
    """

    def __init__(
        self,
        knowledge_path: Optional[str] = None,
        top_k: int = 3,
        min_similarity_threshold: float = 0.15
    ):
        self.top_k = top_k
        self.min_similarity_threshold = min_similarity_threshold
        self.kb = DrishtikonKnowledgeBase(knowledge_path)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        """Builds TF-IDF index over all records' search_text."""
        if not self.kb.is_loaded or len(self.kb.records) == 0:
            logger.info("DRISHTIKON knowledge base is empty; TF-IDF index not built.")
            return

        corpus = [r.get("search_text", "") for r in self.kb.records]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            logger.info(
                "Built DRISHTIKON TF-IDF index for %d records (vocab size: %d)",
                len(corpus), len(self.vectorizer.vocabulary_)
            )
        except ValueError as e:
            # Can happen if corpus is purely non-alphanumeric or empty
            logger.warning("Failed to build TF-IDF index: %s", e)
            self.vectorizer = None
            self.tfidf_matrix = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves top-k relevant DRISHTIKON records for the given query.

        Returns:
            List of dicts, each containing original DRISHTIKON fields + relevance_score.
            Returns empty list if no matches or below similarity threshold.
        """
        if not query or not query.strip():
            return []

        if self.vectorizer is None or self.tfidf_matrix is None or len(self.kb.records) == 0:
            return []

        k = top_k if top_k is not None else self.top_k
        query_clean = query.strip().lower()

        try:
            query_vec = self.vectorizer.transform([query_clean])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Rank indices by score descending
            ranked_indices = sim_scores.argsort()[::-1]

            results = []
            for idx in ranked_indices:
                score = float(sim_scores[idx])
                if score < self.min_similarity_threshold:
                    break  # Since array is sorted descending, remaining scores are even lower

                record = dict(self.kb.records[idx])
                record["relevance_score"] = round(score, 4)
                results.append(record)

                if len(results) >= k:
                    break

            logger.info(
                "DRISHTIKON retrieval for '%s': found %d matches (top score: %s)",
                query[:40], len(results), results[0]["relevance_score"] if results else "N/A"
            )
            return results

        except Exception as e:
            logger.error("Error during DRISHTIKON retrieval: %s", e)
            return []
