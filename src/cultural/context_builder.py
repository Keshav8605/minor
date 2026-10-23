"""
Cultural Context Builder.

Orchestrates cultural retrieval from:
1. Existing Project Cultural Knowledge Base (cultural_knowledge.json)
2. DRISHTIKON External Cultural Dataset (drishtikon_knowledge.json)

RESEARCH INTEGRITY RULES:
- The Context Builder must not transform DRISHTIKON MCQ records into newly generated
  cultural facts or unsupported explanations.
- Any information injected into the VLM prompt must remain directly grounded
  in the retrieved DRISHTIKON question, answer, and metadata.
- Option1–option4 and raw JSON are NEVER injected into the VLM prompt.
- General Mode must remain strictly isolated (never calls ContextBuilder).
- Computes source attribution: PROJECT_JSON, DRISHTIKON, PROJECT_JSON + DRISHTIKON, NONE.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from .context_retriever import CulturalRetriever
from .drishtikon_retriever import DrishtikonRetriever
from .category_detector import detect_categories

logger = logging.getLogger(__name__)

MONTHS_REGEX = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)'
CULTURAL_NUMBERS = {'370', '420', '144', '100', '108', '101', '500', '2000'}

GENERIC_MEME_PREFIXES = [
    r'^\s*me\s+and\s+the\s+boys(?:\s+after|\s+when)?\b',
    r'\bme\s+and\s+the\s+boys\b',
    r'\bnobody\s*:',
    r'\bno\s+one\s*:',
    r'\bpov\s*:',
    r'\ble\s+husband\s*:',
    r'\ble\s+wife\s*:',
    r'\ble\s+me\s*:',
    r'\ble\s*:',
    r'(?i)\bbackup\s*:\s*@\S+',
    r'(?i)\b(?:ig|fb|yt)\s*/\s*[a-zA-Z0-9_.-]+',
    r'(?i)@\w+(?:\.\w+)*',
]


def sanitize_retrieval_query(text: str) -> str:
    """
    Sanitizes raw OCR text for cultural retrieval:
    - Strips generic meme prefixes that add no cultural meaning.
    - Filters isolated numeric noise (e.g. '10,000') that causes false lexical collisions.
    - Preserves meaningful legal/cultural numbers (e.g., Section 144, Article 370, 420),
      calendar dates (e.g., 15 August, 15th Aug), names, and Hinglish phrases.
    """
    if not text or not str(text).strip():
        return ""

    cleaned = str(text)

    # 1. Strip generic meme prefixes
    for pat in GENERIC_MEME_PREFIXES:
        cleaned = re.sub(pat, ' ', cleaned, flags=re.IGNORECASE)

    # 2. Filter isolated numeric tokens, preserving cultural/legal numbers and dates
    def _num_filter(match):
        prefix = match.group(1) or ''
        num_str = match.group(2)
        suffix = match.group(3) or ''
        raw_num = num_str.replace(',', '')

        # Keep if preceded by section/article/act/rule/no
        if prefix.lower().strip() in ('section', 'article', 'act', 'rule', 'no', 'no.', 'dhara', 'dhāra'):
            return match.group(0)
        # Keep if followed by calendar month
        if suffix.strip():
            return match.group(0)
        # Keep culturally significant numbers
        if raw_num in CULTURAL_NUMBERS:
            return match.group(0)
        # Filter out isolated numbers
        return prefix + ' '

    num_pattern = rf'(?i)(\b(?:section|article|act|rule|no\.?|dhara|dhāra)\s+)?\b(\d{{1,3}}(?:,\d{{3}})+|\d+)\b(\s+{MONTHS_REGEX}\b)?'
    cleaned = re.sub(num_pattern, _num_filter, cleaned)

    # 3. Collapse multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class CulturalContextBuilder:
    """
    Unified context builder coordinating project JSON and DRISHTIKON retrievers.
    """

    def __init__(
        self,
        categories_path: str,
        knowledge_path: str,
        drishtikon_path: Optional[str] = None,
        drishtikon_enabled: bool = False,
        drishtikon_top_k: int = 3,
        min_similarity_threshold: float = 0.25
    ):
        self.project_retriever = CulturalRetriever(categories_path, knowledge_path)
        self.drishtikon_enabled = drishtikon_enabled

        if self.drishtikon_enabled and drishtikon_path:
            self.drishtikon_retriever = DrishtikonRetriever(
                knowledge_path=drishtikon_path,
                top_k=drishtikon_top_k,
                min_similarity_threshold=min_similarity_threshold
            )
        else:
            self.drishtikon_retriever = None

    def build_context(self, text: str) -> Tuple[str, List[str], List[Dict]]:
        """
        Retrieves cultural context from active sources for the given text.

        Returns:
            Tuple: (
                prompt_context_str: str (compact text ready for Qwen prompt),
                cultural_sources: List[str] (e.g. ['PROJECT_JSON', 'DRISHTIKON']),
                drishtikon_records: List[Dict] (traceable records with metadata)
            )
        """
        if not text or not text.strip():
            return "", [], []

        sources_used = []
        drishtikon_records = []
        context_sections = []

        # 1. Project Knowledge Retrieval (Existing JSON)
        sanitized_text = sanitize_retrieval_query(text)
        query_text = sanitized_text if sanitized_text else text
        raw_project_context = self.project_retriever.retrieve_context(query_text)
        if (
            raw_project_context
            and raw_project_context.strip()
            and "No specific cultural reference detected" not in raw_project_context
        ):
            sources_used.append("PROJECT_JSON")
            # Extract bullet points from existing retriever output
            lines = raw_project_context.replace("EXTERNAL CULTURAL CONTEXT:\n", "").strip()
            context_sections.append(f"PROJECT CULTURAL KNOWLEDGE:\n{lines}")

        # 2. DRISHTIKON Retrieval (Additive external dataset)
        if self.drishtikon_enabled and self.drishtikon_retriever is not None:
            # Construct query combining sanitized text and detected categories
            detected_cats = detect_categories(text)
            clean_cats = [c for c in detected_cats if c != "none"]
            query_parts = ([sanitized_text] if sanitized_text else [text]) + clean_cats
            query = " ".join(query_parts)

            d_matches = self.drishtikon_retriever.retrieve(query)
            if d_matches:
                sources_used.append("DRISHTIKON")
                drishtikon_records = d_matches

                # Format compact, grounded references WITHOUT injecting option1-option4 or raw JSON
                drishtikon_lines = []
                for rec in d_matches:
                    attr = rec.get("attribute", "General")
                    state = rec.get("state", "India")
                    q = rec.get("question", "")
                    a = rec.get("answer", "")
                    drishtikon_lines.append(f"- [{attr} | {state}] {q} -> Context: {a}")

                context_sections.append(
                    "INDIAN CULTURAL CONTEXT (DRISHTIKON):\n" + "\n".join(drishtikon_lines)
                )

        if not context_sections:
            return "", [], []

        prompt_context = "EXTERNAL CULTURAL CONTEXT:\n" + "\n\n".join(context_sections)
        return prompt_context, sources_used, drishtikon_records
