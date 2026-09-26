"""
Dedicated OCR Engine supporting Hindi (Devanagari) and English text extraction.
Uses EasyOCR when available, with graceful fallback to VLM visual text extraction.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_OCR_READER = None
_OCR_INITIALIZED = False


def get_ocr_reader():
    """Initializes and returns the singleton EasyOCR reader for Hindi and English."""
    global _OCR_READER, _OCR_INITIALIZED
    if _OCR_INITIALIZED:
        return _OCR_READER

    _OCR_INITIALIZED = True
    try:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        logger.info("Initializing EasyOCR reader (Hindi + English, gpu=%s)...", gpu)
        _OCR_READER = easyocr.Reader(['hi', 'en'], gpu=gpu)
        logger.info("EasyOCR reader initialized successfully.")
    except Exception as e:
        logger.warning("Could not initialize dedicated EasyOCR reader: %s. Will fallback to VLM visual text understanding.", e)
        _OCR_READER = None

    return _OCR_READER


def extract_text_ocr(image_path: str) -> tuple:
    """
    Extracts text from an image using dedicated OCR.

    Returns:
        tuple: (extracted_text: str, engine_name: str)
               engine_name is 'EasyOCR (Hindi+English)' or 'None'.
    """
    reader = get_ocr_reader()
    if reader is None:
        return "", "None"

    try:
        path = str(Path(image_path).resolve())
        if not os.path.exists(path):
            return "", "None"

        results = reader.readtext(path)
        # results format: [ (bbox, text, prob), ... ]
        text_pieces = [res[1].strip() for res in results if res and len(res) > 1 and res[1].strip()]
        combined_text = " ".join(text_pieces)
        logger.info("[OCR] Dedicated OCR extracted %d segments: %s", len(text_pieces), combined_text[:150])
        return combined_text, "EasyOCR (Hindi+English)"
    except Exception as e:
        logger.warning("OCR extraction failed on %s: %s", image_path, e)
        return "", "None"
