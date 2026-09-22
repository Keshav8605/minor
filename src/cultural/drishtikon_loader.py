"""
DRISHTIKON Dataset Loader.

Loads and parses locally prepared DRISHTIKON cultural knowledge records.

The 12 official DRISHTIKON fields are:
    language, state, attribute, question_type, question,
    option1, option2, option3, option4, answer,
    image_name, image_link.

'id' and 'search_text' are derived processing fields created by our pipeline
and are NOT official DRISHTIKON fields. The 12 official fields are preserved unchanged.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

OFFICIAL_FIELDS = [
    "language", "state", "attribute", "question_type", "question",
    "option1", "option2", "option3", "option4", "answer",
    "image_name", "image_link"
]


class DrishtikonKnowledgeBase:
    """
    In-memory knowledge base of processed DRISHTIKON records.
    Provides safe loading and access without mutating or fabricating records.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None
        self.records: List[Dict] = []
        self.is_loaded: bool = False

        if self.data_path and self.data_path.exists():
            self.load(self.data_path)

    def load(self, path: Path) -> int:
        """Loads records from the processed JSON file."""
        self.data_path = Path(path)
        if not self.data_path.exists():
            logger.warning("DRISHTIKON knowledge file not found at: %s", self.data_path)
            self.records = []
            self.is_loaded = False
            return 0

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, list):
                logger.error("Expected list of records in %s, got %s", self.data_path, type(raw_data))
                self.records = []
                self.is_loaded = False
                return 0

            self.records = raw_data
            self.is_loaded = True
            logger.info("Loaded %d DRISHTIKON records from %s", len(self.records), self.data_path)
            return len(self.records)

        except Exception as e:
            logger.error("Failed to load DRISHTIKON knowledge base from %s: %s", self.data_path, e)
            self.records = []
            self.is_loaded = False
            return 0

    def get_record_by_id(self, record_id: int) -> Optional[Dict]:
        """Retrieves a single record by its derived integer id."""
        if 0 <= record_id < len(self.records):
            return self.records[record_id]
        return None

    def __len__(self) -> int:
        return len(self.records)
