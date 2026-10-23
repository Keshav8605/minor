"""
DRISHTIKON Cultural Context Retriever.

Uses TF-IDF vectorization and cosine similarity over deterministically constructed
'search_text' to retrieve the most relevant DRISHTIKON cultural records.

Preserves full source traceability (source, id, state, attribute, language,
question, answer, image_name, image_link, relevance_score).

Does NOT fabricate explanations or invent facts.
"""

import re
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from .drishtikon_loader import DrishtikonKnowledgeBase

logger = logging.getLogger(__name__)

# Generic structural tokens in QA/metadata that must not count as meaningful content tokens
GENERIC_STRUCTURAL = {"section", "depicted", "image", "question", "answer", "option", "general"}

# Indian states and territories to separate geographic anchors from core cultural content
INDIAN_STATES = {
    "delhi", "haryana", "punjab", "uttarakhand", "bihar", "gujarat",
    "maharashtra", "karnataka", "tamil_nadu", "tamil", "nadu",
    "west_bengal", "bengal", "kerala", "rajasthan", "odisha",
    "jharkhand", "telangana", "uttar_pradesh", "madhya_pradesh",
    "assam", "sikkim", "puducherry", "chandigarh", "goa", "india", "indian"
}


def _extract_content_tokens(text: str) -> set:
    """Extracts non-stopword, non-state, non-structural content tokens."""
    tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    return tokens - ENGLISH_STOP_WORDS - INDIAN_STATES - GENERIC_STRUCTURAL


def _extract_state_tokens(text: str) -> set:
    """Extracts Indian state and territory tokens."""
    tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
    return tokens & INDIAN_STATES


class DrishtikonRetriever:
    """
    Lightweight, deterministic TF-IDF retriever for DRISHTIKON records.
    """

    def __init__(
        self,
        knowledge_path: Optional[str] = None,
        top_k: int = 3,
        min_similarity_threshold: float = 0.25
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

            query_tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", query_clean))
            query_content = _extract_content_tokens(query_clean)
            query_states = _extract_state_tokens(query_clean)
            query_has_structural = bool(query_tokens & GENERIC_STRUCTURAL)

            results = []
            for idx in ranked_indices:
                score = float(sim_scores[idx])
                if score < self.min_similarity_threshold:
                    break  # Since array is sorted descending, remaining scores are even lower

                record = dict(self.kb.records[idx])

                # Content-token overlap constraint for weak matches below 0.45
                if score < 0.45:
                    doc_search_text = record.get("search_text", "").lower()
                    doc_content = _extract_content_tokens(doc_search_text)
                    content_overlap = query_content.intersection(doc_content)

                    doc_states = _extract_state_tokens(doc_search_text)
                    state_overlap = query_states.intersection(doc_states)

                    # For queries containing generic structural tokens (e.g. 'section'), require multi-token evidence:
                    # at least 2 distinct non-state content tokens, OR at least 1 non-state content token + state agreement.
                    # For standard queries, require at least 1 meaningful non-state, non-structural content token.
                    if query_has_structural:
                        has_sufficient_overlap = (len(content_overlap) >= 2) or (
                            len(content_overlap) >= 1 and len(state_overlap) >= 1
                        )
                    else:
                        has_sufficient_overlap = len(content_overlap) >= 1

                    if not has_sufficient_overlap:
                        continue

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
