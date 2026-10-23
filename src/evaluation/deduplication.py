"""
Pure PIL and NumPy Image & Text Deduplication Utilities.

Provides zero-dependency perceptual hashing (average hash / difference hash)
and normalized OCR similarity matching to prevent data leakage between historical
evaluations and held-out test splits.
"""

import os
import re
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from PIL import Image


def compute_file_md5(filepath: Union[str, Path]) -> str:
    """Computes exact MD5 checksum of an image file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_average_hash(image_input: Union[str, Path, Image.Image], hash_size: int = 8) -> Optional[np.ndarray]:
    """
    Computes an 8x8 perceptual average hash (aHash) using pure PIL and NumPy.
    
    Algorithm:
    1. Convert image to grayscale.
    2. Resize to (hash_size, hash_size) using high-quality Lanczos resampling.
    3. Compute mean intensity across all pixels.
    4. Produce a binary bit array where bit = 1 if pixel > mean else 0.
    
    Returns:
        np.ndarray of shape (hash_size * hash_size,) dtype=bool, or None if image cannot be opened.
    """
    try:
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
                pixels = np.array(img, dtype=np.float32)
        else:
            img = image_input.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
        
        avg = pixels.mean()
        return (pixels > avg).flatten()
    except Exception:
        return None


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Computes the Hamming distance (number of bit differences) between two perceptual hashes."""
    if hash1 is None or hash2 is None:
        return 999
    return int(np.count_nonzero(hash1 != hash2))


def normalize_ocr_text(text: str) -> str:
    """Normalizes OCR text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_ocr_similarity(text1: str, text2: str) -> float:
    """Computes Levenshtein-based sequence similarity ratio between normalized OCR texts."""
    norm1 = normalize_ocr_text(text1)
    norm2 = normalize_ocr_text(text2)
    if not norm1 or not norm2:
        return 0.0
    return SequenceMatcher(None, norm1, norm2).ratio()


def check_candidate_against_quarantine(
    candidate_img_path: Union[str, Path],
    candidate_ocr: str,
    quarantined_data: List[dict],
    max_hamming_distance: int = 3,
    min_ocr_similarity: float = 0.85
) -> Optional[dict]:
    """
    Checks a single candidate meme against a list of quarantined reference memes.
    
    quarantined_data must be a list of dicts with:
      {"filename": str, "md5": str, "ahash": np.ndarray, "ocr": str}
      
    Returns:
        dict with collision details if a duplicate is found, else None.
    """
    c_path = Path(candidate_img_path)
    if not c_path.exists():
        return None
        
    c_md5 = compute_file_md5(c_path)
    c_ahash = compute_average_hash(c_path)
    c_ocr = candidate_ocr or ""
    
    for q in quarantined_data:
        # Check 1: Exact MD5
        if q.get("md5") and c_md5 == q["md5"]:
            return {
                "collision_type": "EXACT_MD5",
                "quarantined_file": q["filename"],
                "distance": 0,
            }
            
        # Check 2: Perceptual aHash
        if c_ahash is not None and q.get("ahash") is not None:
            dist = hamming_distance(c_ahash, q["ahash"])
            if dist <= max_hamming_distance:
                return {
                    "collision_type": "PERCEPTUAL_AHASH",
                    "quarantined_file": q["filename"],
                    "distance": dist,
                }
                
        # Check 3: Normalized OCR similarity
        if len(normalize_ocr_text(c_ocr)) >= 15 and len(normalize_ocr_text(q.get("ocr", ""))) >= 15:
            sim = compute_ocr_similarity(c_ocr, q["ocr"])
            if sim >= min_ocr_similarity:
                return {
                    "collision_type": "OCR_SIMILARITY",
                    "quarantined_file": q["filename"],
                    "distance": round(sim, 4),
                }
                
    return None
