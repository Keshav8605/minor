"""
Download and preparation script for the DRISHTIKON Indian-culture dataset.

Dataset Source: https://huggingface.co/datasets/13ari/DRISHTIKON
Paper: arXiv:2509.19274

The 12 official DRISHTIKON fields are:
    language, state, attribute, question_type, question,
    option1, option2, option3, option4, answer,
    image_name, image_link.

'id' and 'search_text' are derived processing fields created by our pipeline
and are NOT official DRISHTIKON fields. The 12 official fields are preserved unchanged.

'search_text' is generated deterministically using only the original DRISHTIKON fields.
It does not involve LLM rewriting, summarization, inference, paraphrasing,
external knowledge, or fabricated cultural information.
"""

import os
import sys
import csv
import json
import logging
import argparse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_drishtikon")

OFFICIAL_FIELDS = [
    "language", "state", "attribute", "question_type", "question",
    "option1", "option2", "option3", "option4", "answer",
    "image_name", "image_link"
]

DEFAULT_CSV_URL = (
    "https://huggingface.co/datasets/13ari/DRISHTIKON/resolve/main/"
    "Drishtikon_all_languages_and_qtypes_merged_dataset.csv"
)
DEFAULT_RAW_DIR = Path("data/cultural/drishtikon/raw")
DEFAULT_PROCESSED_PATH = Path("data/cultural/drishtikon/processed/drishtikon_knowledge.json")


def build_deterministic_search_text(record: dict) -> str:
    """
    Constructs search_text strictly and deterministically from original fields.
    Uses only: question, answer, attribute, state, language, question_type, image_name.
    """
    parts = [
        str(record.get("question", "")).strip(),
        str(record.get("answer", "")).strip(),
        str(record.get("attribute", "")).strip(),
        str(record.get("state", "")).strip(),
        str(record.get("language", "")).strip(),
        str(record.get("question_type", "")).strip(),
        str(record.get("image_name", "")).strip(),
    ]
    return " ".join(p.lower() for p in parts if p)


def download_raw_csv(url: str, dest_path: Path, max_records: int = None) -> Path:
    """Downloads the official DRISHTIKON CSV from Hugging Face."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading official DRISHTIKON CSV from: %s", url)
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DRISHTIKON-Cultural-Humor-Research/1.0"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 64 * 1024
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if downloaded % (1024 * 1024 * 5) < chunk_size:
                logger.info("Downloaded %d MB...", downloaded // (1024 * 1024))

    logger.info("Raw CSV saved successfully to: %s (%d bytes)", dest_path, dest_path.stat().st_size)
    return dest_path


def process_drishtikon_csv(csv_path: Path, output_json_path: Path, max_records: int = None) -> int:
    """
    Parses the raw DRISHTIKON CSV, validates the 12 official fields,
    adds the 2 derived processing fields (id, search_text), and exports JSON.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {csv_path}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    processed_records = []

    logger.info("Processing DRISHTIKON CSV: %s", csv_path)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Validate that question and answer exist
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if not q or not a:
                continue

            # Preserve all 12 official fields
            clean_record = {}
            for field in OFFICIAL_FIELDS:
                clean_record[field] = str(row.get(field, "")).strip()

            # Add derived processing fields
            clean_record["id"] = len(processed_records)
            clean_record["search_text"] = build_deterministic_search_text(clean_record)

            processed_records.append(clean_record)

            if max_records and len(processed_records) >= max_records:
                logger.info("Reached requested record limit: %d", max_records)
                break

    logger.info("Processed %d valid DRISHTIKON records.", len(processed_records))

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    logger.info("Saved processed knowledge dataset to: %s", output_json_path)
    return len(processed_records)


def main():
    parser = argparse.ArgumentParser(description="Download & prepare DRISHTIKON Indian-culture dataset")
    parser.add_argument("--url", default=DEFAULT_CSV_URL, help="URL to official DRISHTIKON CSV")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory for raw download")
    parser.add_argument("--output", default=str(DEFAULT_PROCESSED_PATH), help="Output JSON path")
    parser.add_argument("--max-records", type=int, default=None, help="Optional maximum records to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if raw CSV already exists")
    args = parser.parse_args()

    raw_csv_path = Path(args.raw_dir) / "Drishtikon_all_languages_and_qtypes_merged_dataset.csv"
    if not args.skip_download or not raw_csv_path.exists():
        download_raw_csv(args.url, raw_csv_path)

    output_path = Path(args.output)
    process_drishtikon_csv(raw_csv_path, output_path, max_records=args.max_records)
    logger.info("DRISHTIKON preparation complete.")


if __name__ == "__main__":
    main()
