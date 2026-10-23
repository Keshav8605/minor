"""
Held-Out Evaluation Manifest Generator for Drishtikon Project.

Creates a stratified, balanced (50 humorous / 50 non-humorous) evaluation benchmark of 100 memes
from data/processed/test.csv, strictly enforcing:
1. Complete exclusion of the 20 historical evaluation memes.
2. Complete exclusion of perceptual hash collisions (aHash d <= 3).
3. Complete exclusion of normalized OCR text collisions (> 85% Levenshtein similarity).
4. Deterministic sampling via fixed seed (default: 42).
5. Exact quota fulfillment across all 11 strata (H1-H5, NH1-NH6).
6. Immutable manifest format saved for multi-configuration benchmarking.

Zero VLM inference is invoked.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

import pandas as pd
from PIL import Image

from src.evaluation.deduplication import (
    compute_file_md5,
    compute_average_hash,
    hamming_distance,
    normalize_ocr_text,
    compute_ocr_similarity,
    check_candidate_against_quarantine,
)
from src.cultural.category_detector import CATEGORY_KEYWORDS


# ─── 20 Historical Quarantined Memes ─────────────────────────────────────────
QUARANTINED_HISTORICAL_MEMES: Set[str] = {
    "train_55.jpg", "train_127.jpg", "train_833.jpg", "train_1611.jpg",
    "train_2000.jpg", "train_2183.jpg", "train_2573.jpg", "train_2837.jpg",
    "train_3280.jpg", "train_3395.jpg", "train_4020.jpg", "train_4136.jpg",
    "train_4199.jpg", "train_4917.jpg", "train_5088.jpg", "train_5266.jpg",
    "train_5550.jpg", "train_5723.jpg", "train_5890.jpg", "train_6346.jpg"
}

# ─── Quota Specifications ───────────────────────────────────────────────────
STRATA_QUOTAS: Dict[str, Dict[str, Any]] = {
    # Humorous Class (Total = 50)
    "H1": {
        "class": "humorous",
        "name": "High Cultural Dependency",
        "quota": 15,
        "description": "Bollywood, cricket/IPL, festivals, joint family, arranged marriage"
    },
    "H2": {
        "class": "humorous",
        "name": "Hinglish Daily Life & Situational Irony",
        "quota": 15,
        "description": "Latin-script Hindi daily life banter and relatable middle-class tropes"
    },
    "H3": {
        "class": "humorous",
        "name": "Political Satire & Societal Commentary",
        "quota": 10,
        "description": "Subtle rhetorical irony, governance, public affairs"
    },
    "H4": {
        "class": "humorous",
        "name": "Dark Humour & Edgy Comedy",
        "quota": 5,
        "description": "Offensive/edgy comedic mechanisms and taboo subjects"
    },
    "H5": {
        "class": "humorous",
        "name": "Universal / Non-Cultural Situational Comedy",
        "quota": 5,
        "description": "Visual slapstick, animal memes, workplace comedy without Indian cultural entities"
    },
    # Non-Humorous Class (Total = 50)
    "NH1": {
        "class": "non_humorous",
        "name": "Serious News, Tragedy, & Current Affairs Juxtaposition",
        "quota": 12,
        "description": "Real news events, COVID updates, tragedy formatted in meme-like cards"
    },
    "NH2": {
        "class": "non_humorous",
        "name": "Motivational Quotes & Inspirational Sayings",
        "quota": 12,
        "description": "Quotes, philosophical sayings, moral messages without comedy"
    },
    "NH3": {
        "class": "non_humorous",
        "name": "Factual, Educational, & Historical Graphics",
        "quota": 4,
        "description": "Infographics, charts, timelines, science facts, civic announcements"
    },
    "NH4": {
        "class": "non_humorous",
        "name": "Sarcastic Non-Humour (Subtle/Cynical)",
        "quota": 10,
        "description": "Biting cynicism, sarcastic tone but rated non-humorous by consensus"
    },
    "NH5": {
        "class": "non_humorous",
        "name": "Hostile / Aggressive Social Media Statements",
        "quota": 6,
        "description": "Polarized political or personal hostility without wit or punchline"
    },
    "NH6": {
        "class": "non_humorous",
        "name": "Commercial Advertisements & Watermarked Spam",
        "quota": 6,
        "description": "Promotional product flyers, health spam, telegram channels"
    }
}


def build_quarantined_reference_data(
    test_csv: pd.DataFrame,
    image_dir: Path
) -> List[Dict[str, Any]]:
    """Builds reference hash and OCR data for the 20 historical evaluation memes."""
    quarantined_data = []
    for fn in sorted(list(QUARANTINED_HISTORICAL_MEMES)):
        img_path = image_dir / fn
        if not img_path.exists():
            raise FileNotFoundError(f"Quarantined historical image not found: {img_path}")
        
        row_match = test_csv[test_csv["image_filename"] == fn]
        ocr = str(row_match.iloc[0]["ocr"]) if len(row_match) > 0 and pd.notna(row_match.iloc[0]["ocr"]) else ""
        
        md5 = compute_file_md5(img_path)
        ahash = compute_average_hash(img_path)
        if ahash is None:
            raise ValueError(f"Failed to compute aHash for quarantined historical image: {img_path}")
            
        quarantined_data.append({
            "filename": fn,
            "path": str(img_path),
            "md5": md5,
            "ahash": ahash,
            "ocr": ocr
        })
    return quarantined_data


def filter_candidate_pool(
    test_csv: pd.DataFrame,
    image_dir: Path,
    quarantined_data: List[Dict[str, Any]],
    max_hamming_distance: int = 3,
    min_ocr_similarity: float = 0.85
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Applies strict deduplication against quarantined historical evaluation memes.
    Returns (clean_candidates_df, list_of_exclusions).
    """
    exclusions = []
    
    # 1. Filename exclusion
    candidates = test_csv[~test_csv["image_filename"].isin(QUARANTINED_HISTORICAL_MEMES)].copy()
    for fn in QUARANTINED_HISTORICAL_MEMES:
        exclusions.append({
            "candidate_file": fn,
            "exclusion_reason": "HISTORICAL_EVALUATION_SAMPLE",
            "collision_type": "EXACT_FILENAME",
            "reference_file": fn,
            "metric_value": 0
        })
        
    clean_indices = []
    
    for idx, row in candidates.iterrows():
        fn = row["image_filename"]
        img_path = image_dir / fn
        
        if not img_path.exists():
            exclusions.append({
                "candidate_file": fn,
                "exclusion_reason": "MISSING_IMAGE_FILE",
                "collision_type": "FILE_NOT_FOUND",
                "reference_file": None,
                "metric_value": None
            })
            continue
            
        # Verify image readability
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            exclusions.append({
                "candidate_file": fn,
                "exclusion_reason": "UNREADABLE_IMAGE_FILE",
                "collision_type": "CORRUPT_IMAGE",
                "reference_file": None,
                "metric_value": str(e)
            })
            continue
            
        ocr = str(row["ocr"]) if pd.notna(row["ocr"]) else ""
        collision = check_candidate_against_quarantine(
            img_path,
            ocr,
            quarantined_data,
            max_hamming_distance=max_hamming_distance,
            min_ocr_similarity=min_ocr_similarity
        )
        
        if collision:
            exclusions.append({
                "candidate_file": fn,
                "exclusion_reason": f"COLLISION_WITH_HISTORICAL_MEME ({collision['collision_type']})",
                "collision_type": collision["collision_type"],
                "reference_file": collision["quarantined_file"],
                "metric_value": collision["distance"]
            })
        else:
            clean_indices.append(idx)
            
    clean_candidates_df = candidates.loc[clean_indices].reset_index(drop=True)
    return clean_candidates_df, exclusions


def sample_heldout_manifest(
    clean_df: pd.DataFrame,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Deterministically samples exactly 100 memes across 11 strata using seed.
    """
    clean_nh = clean_df[clean_df["is_humorous"] == 0].sort_values("image_filename").reset_index(drop=True)
    clean_h = clean_df[clean_df["is_humorous"] == 1].sort_values("image_filename").reset_index(drop=True)
    
    rng = random.Random(seed)
    selected_samples: List[Dict[str, Any]] = []
    used_filenames: Set[str] = set()
    
    # ── Non-Humorous Sampling ──
    spam_terms = [
        'herbalife', 'call', 'order', 'brand', 'offer', 'troll', 'advertisement',
        't.me', 'follow @', 'telegram', 'nutrition', 'deal', 'discount', 'free shipping', 'shop now'
    ]
    factual_terms = [
        'mobile application users', 'a fifth of indians', 'percent', '%', 'study',
        'research', 'timeline', 'did you know', 'infographic', 'facts', 'science', 'statistics', 'survey'
    ]
    news_terms = [
        'corona', 'covid', 'modi', 'govt', 'government', 'police', 'constitution',
        'delhi', 'court', 'news', 'minister', 'protest', 'curfew', 'alien', 'lockdown',
        'tragedy', 'tragic', 'immigration', 'suffering', 'murder', 'arrest', 'crime', 'border', 'soldier', 'army', 'death'
    ]
    
    # Priority 1: NH3 (Factual/Educational) - quota: 4
    nh3_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and any(t in str(r["ocr"]).lower() for t in factual_terms)
    ]
    if len(nh3_pool) < STRATA_QUOTAS["NH3"]["quota"]:
        raise ValueError(f"Insufficient NH3 candidates: found {len(nh3_pool)}, required {STRATA_QUOTAS['NH3']['quota']}")
    sampled_nh3 = rng.sample(nh3_pool, STRATA_QUOTAS["NH3"]["quota"])
    for r in sampled_nh3:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH3", "row": r})
        
    # Priority 2: NH6 (Commercial Spam) - quota: 6
    nh6_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and any(t in str(r["ocr"]).lower() for t in spam_terms)
    ]
    if len(nh6_pool) < STRATA_QUOTAS["NH6"]["quota"]:
        raise ValueError(f"Insufficient NH6 candidates: found {len(nh6_pool)}, required {STRATA_QUOTAS['NH6']['quota']}")
    sampled_nh6 = rng.sample(nh6_pool, STRATA_QUOTAS["NH6"]["quota"])
    for r in sampled_nh6:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH6", "row": r})
        
    # Priority 3: NH5 (Hostile Statements) - quota: 6
    nh5_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and r["offensive"] in ['very_offensive', 'hateful_offensive']
    ]
    if len(nh5_pool) < STRATA_QUOTAS["NH5"]["quota"]:
        raise ValueError(f"Insufficient NH5 candidates: found {len(nh5_pool)}, required {STRATA_QUOTAS['NH5']['quota']}")
    sampled_nh5 = rng.sample(nh5_pool, STRATA_QUOTAS["NH5"]["quota"])
    for r in sampled_nh5:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH5", "row": r})
        
    # Priority 4: NH1 (Serious News) - quota: 12
    nh1_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and any(t in str(r["ocr"]).lower() for t in news_terms)
    ]
    if len(nh1_pool) < STRATA_QUOTAS["NH1"]["quota"]:
        raise ValueError(f"Insufficient NH1 candidates: found {len(nh1_pool)}, required {STRATA_QUOTAS['NH1']['quota']}")
    sampled_nh1 = rng.sample(nh1_pool, STRATA_QUOTAS["NH1"]["quota"])
    for r in sampled_nh1:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH1", "row": r})
        
    # Priority 5: NH2 (Motivational) - quota: 12
    nh2_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and r["motivational"] == 'motivational'
    ]
    if len(nh2_pool) < STRATA_QUOTAS["NH2"]["quota"]:
        raise ValueError(f"Insufficient NH2 candidates: found {len(nh2_pool)}, required {STRATA_QUOTAS['NH2']['quota']}")
    sampled_nh2 = rng.sample(nh2_pool, STRATA_QUOTAS["NH2"]["quota"])
    for r in sampled_nh2:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH2", "row": r})
        
    # Priority 6: NH4 (Sarcastic Non-Humour) - quota: 10
    nh4_pool = [
        r for _, r in clean_nh.iterrows()
        if r["image_filename"] not in used_filenames and r["sarcastic"] in ['twisted_meaning', 'very_twisted', 'general']
    ]
    if len(nh4_pool) < STRATA_QUOTAS["NH4"]["quota"]:
        raise ValueError(f"Insufficient NH4 candidates: found {len(nh4_pool)}, required {STRATA_QUOTAS['NH4']['quota']}")
    sampled_nh4 = rng.sample(nh4_pool, STRATA_QUOTAS["NH4"]["quota"])
    for r in sampled_nh4:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "NH4", "row": r})
        
    # ── Humorous Sampling ──
    cultural_kw = set()
    for cat, kw_list in CATEGORY_KEYWORDS.items():
        cultural_kw.update([k.lower() for k in kw_list])
    hinglish_markers = [
        'bhai', 'yaar', 'kya', 'nahi', 'karo', 'meri', 'mera', 'tera', 'teri', 'hum',
        'tum', 'padhai', 'baat', 'wala', 'wali', 'karte', 'karta', 'dekh', 'dekho',
        'kuch', 'hoga', 'raha', 'rahi', 'hai', 'hain', 'kaise', 'sab', 'apne', 'beta', 'padosi', 'jugaad', 'arey'
    ]
    
    # Priority 1: H5 (Universal/Non-Cultural) - quota: 5
    h5_pool = [
        r for _, r in clean_h.iterrows()
        if r["image_filename"] not in used_filenames
        and not any(k in str(r["ocr"]).lower() for k in cultural_kw)
        and not any(m in str(r["ocr"]).lower() for m in hinglish_markers)
    ]
    if len(h5_pool) < STRATA_QUOTAS["H5"]["quota"]:
        raise ValueError(f"Insufficient H5 candidates: found {len(h5_pool)}, required {STRATA_QUOTAS['H5']['quota']}")
    sampled_h5 = rng.sample(h5_pool, STRATA_QUOTAS["H5"]["quota"])
    for r in sampled_h5:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "H5", "row": r})
        
    # Priority 2: H4 (Dark Humour & Edgy) - quota: 5
    h4_pool = [
        r for _, r in clean_h.iterrows()
        if r["image_filename"] not in used_filenames and r["offensive"] in ['slight', 'very_offensive', 'hateful_offensive']
    ]
    if len(h4_pool) < STRATA_QUOTAS["H4"]["quota"]:
        raise ValueError(f"Insufficient H4 candidates: found {len(h4_pool)}, required {STRATA_QUOTAS['H4']['quota']}")
    sampled_h4 = rng.sample(h4_pool, STRATA_QUOTAS["H4"]["quota"])
    for r in sampled_h4:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "H4", "row": r})
        
    # Priority 3: H3 (Political Satire) - quota: 10
    h3_pool = [
        r for _, r in clean_h.iterrows()
        if r["image_filename"] not in used_filenames and r["sarcastic"] in ['twisted_meaning', 'very_twisted']
    ]
    if len(h3_pool) < STRATA_QUOTAS["H3"]["quota"]:
        raise ValueError(f"Insufficient H3 candidates: found {len(h3_pool)}, required {STRATA_QUOTAS['H3']['quota']}")
    sampled_h3 = rng.sample(h3_pool, STRATA_QUOTAS["H3"]["quota"])
    for r in sampled_h3:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "H3", "row": r})
        
    # Priority 4: H1 (High Cultural Dependency) - quota: 15
    h1_pool = [
        r for _, r in clean_h.iterrows()
        if r["image_filename"] not in used_filenames and any(k in str(r["ocr"]).lower() for k in cultural_kw)
    ]
    if len(h1_pool) < STRATA_QUOTAS["H1"]["quota"]:
        raise ValueError(f"Insufficient H1 candidates: found {len(h1_pool)}, required {STRATA_QUOTAS['H1']['quota']}")
    sampled_h1 = rng.sample(h1_pool, STRATA_QUOTAS["H1"]["quota"])
    for r in sampled_h1:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "H1", "row": r})
        
    # Priority 5: H2 (Hinglish Daily Life) - quota: 15
    h2_pool = [
        r for _, r in clean_h.iterrows()
        if r["image_filename"] not in used_filenames and any(m in str(r["ocr"]).lower() for m in hinglish_markers)
    ]
    if len(h2_pool) < STRATA_QUOTAS["H2"]["quota"]:
        raise ValueError(f"Insufficient H2 candidates: found {len(h2_pool)}, required {STRATA_QUOTAS['H2']['quota']}")
    sampled_h2 = rng.sample(h2_pool, STRATA_QUOTAS["H2"]["quota"])
    for r in sampled_h2:
        used_filenames.add(r["image_filename"])
        selected_samples.append({"stratum_id": "H2", "row": r})
        
    if len(selected_samples) != 100:
        raise ValueError(f"Sampling produced {len(selected_samples)} items instead of exactly 100")
        
    # Deterministic interleave / shuffle using seed
    rng.shuffle(selected_samples)
    return selected_samples


def build_manifest_document(
    selected_samples: List[Dict[str, Any]],
    image_dir: Path,
    exclusions: List[Dict[str, Any]],
    clean_candidates_count: int,
    total_source_count: int,
    seed: int = 42
) -> Dict[str, Any]:
    """Assembles the final immutable JSON manifest structure."""
    manifest_entries = []
    
    for idx, item in enumerate(selected_samples):
        r = item["row"]
        stratum_id = item["stratum_id"]
        stratum_info = STRATA_QUOTAS[stratum_id]
        
        fn = r["image_filename"]
        img_path = image_dir / fn
        md5_val = compute_file_md5(img_path)
        ahash_arr = compute_average_hash(img_path)
        ahash_hex = "".join(["1" if b else "0" for b in ahash_arr]) if ahash_arr is not None else ""
        
        ocr_text = str(r["ocr"]) if pd.notna(r["ocr"]) else ""
        
        # Ambiguity review flag according to section 5.3 of evaluation plan
        # Known edge-case patterns get flagged as AMBIGUOUS_BORDERLINE for disaggregated reporting
        ambiguity_status = "CLEAN_CONSENSUS"
        ambiguity_reason = None
        
        # Borderline pattern 1: Tragic news or serious political controversy in NH1
        if stratum_id == "NH1" and any(k in ocr_text.lower() for k in ["corona", "covid", "death", "tragic", "curfew"]):
            ambiguity_status = "AMBIGUOUS_BORDERLINE"
            ambiguity_reason = "Tragic news or health advisory formatted in meme card template"
        # Borderline pattern 2: Sarcastic Non-Humour in NH4
        elif stratum_id == "NH4" and r["sarcastic"] in ["twisted_meaning", "very_twisted"]:
            ambiguity_status = "AMBIGUOUS_BORDERLINE"
            ambiguity_reason = "Biting cynicism or dark sarcasm labelled not_funny by crowd annotators"
        # Borderline pattern 3: Dark comedy in H4
        elif stratum_id == "H4" and r["offensive"] in ["very_offensive", "hateful_offensive"]:
            ambiguity_status = "AMBIGUOUS_BORDERLINE"
            ambiguity_reason = "Edgy or taboo comedy with potential moral disapproval from evaluators"
            
        entry = {
            "sample_id": f"heldout_{idx + 1:03d}",
            "sample_index": idx,
            "image_filename": fn,
            "relative_image_path": f"data/processed/images/{fn}",
            "ground_truth": int(r["is_humorous"]),
            "is_humorous": int(r["is_humorous"]),
            "class_name": stratum_info["class"],
            "stratum_id": stratum_id,
            "stratum_name": stratum_info["name"],
            "stratum_description": stratum_info["description"],
            "source_dataset": "Memotion 3 (De-Factify 2, AAAI 2023)",
            "source_split": "data/processed/test.csv",
            "original_labels": {
                "humour": str(r["humour"]),
                "sarcastic": str(r["sarcastic"]),
                "offensive": str(r["offensive"]),
                "motivational": str(r["motivational"]),
                "overall": str(r["overall"]) if "overall" in r else None
            },
            "ocr_text": ocr_text,
            "file_md5": md5_val,
            "ahash_bitstring": ahash_hex,
            "deduplication_status": {
                "quarantine_overlap": False,
                "perceptual_ahash_collision": False,
                "ocr_similarity_collision": False,
                "verified": True
            },
            "ambiguity_status": ambiguity_status,
            "ambiguity_reason": ambiguity_reason
        }
        manifest_entries.append(entry)
        
    strata_counts = {}
    for entry in manifest_entries:
        sid = entry["stratum_id"]
        strata_counts[sid] = strata_counts.get(sid, 0) + 1
        
    humorous_count = sum(1 for e in manifest_entries if e["ground_truth"] == 1)
    non_humorous_count = sum(1 for e in manifest_entries if e["ground_truth"] == 0)
    clean_consensus_count = sum(1 for e in manifest_entries if e["ambiguity_status"] == "CLEAN_CONSENSUS")
    borderline_count = sum(1 for e in manifest_entries if e["ambiguity_status"] == "AMBIGUOUS_BORDERLINE")
    
    manifest_doc = {
        "manifest_version": "1.0.0",
        "created_at": "2026-10-20",
        "random_seed": seed,
        "sample_size": len(manifest_entries),
        "target_balance": {
            "humorous": humorous_count,
            "non_humorous": non_humorous_count
        },
        "ambiguity_breakdown": {
            "clean_consensus_samples": clean_consensus_count,
            "ambiguous_borderline_samples": borderline_count
        },
        "strata_counts": strata_counts,
        "strata_quotas": {k: v["quota"] for k, v in STRATA_QUOTAS.items()},
        "candidate_pool_audit": {
            "source_csv": "data/processed/test.csv",
            "total_source_records": total_source_count,
            "total_clean_candidates": clean_candidates_count,
            "total_exclusions": len(exclusions),
            "historical_quarantined_count": len(QUARANTINED_HISTORICAL_MEMES),
            "exclusions_summary": {
                "historical_quarantined": sum(1 for e in exclusions if e["exclusion_reason"] == "HISTORICAL_EVALUATION_SAMPLE"),
                "perceptual_collisions": sum(1 for e in exclusions if "PERCEPTUAL_AHASH" in e["collision_type"]),
                "ocr_collisions": sum(1 for e in exclusions if "OCR_SIMILARITY" in e["collision_type"]),
                "missing_or_corrupt": sum(1 for e in exclusions if e["collision_type"] in ["FILE_NOT_FOUND", "CORRUPT_IMAGE"])
            }
        },
        "samples": manifest_entries
    }
    return manifest_doc


def validate_manifest(
    manifest_doc: Dict[str, Any],
    image_dir: Path,
    quarantined_data: List[Dict[str, Any]]
) -> List[str]:
    """
    Exhaustively validates the manifest document against all 12 safety criteria.
    Returns list of error messages (empty if valid).
    """
    errors = []
    samples = manifest_doc.get("samples", [])
    
    # 1. Exactly 100 unique samples
    if len(samples) != 100:
        errors.append(f"Sample count must be exactly 100, got {len(samples)}")
        
    seen_ids = set()
    seen_files = set()
    seen_paths = set()
    
    h_count = 0
    nh_count = 0
    strata_counts = {k: 0 for k in STRATA_QUOTAS}
    
    for s in samples:
        sid = s.get("sample_id")
        fn = s.get("image_filename")
        rel_path = s.get("relative_image_path")
        gt = s.get("ground_truth")
        stratum = s.get("stratum_id")
        
        # 5. No duplicate sample IDs or paths
        if sid in seen_ids:
            errors.append(f"Duplicate sample_id: {sid}")
        seen_ids.add(sid)
        
        if fn in seen_files:
            errors.append(f"Duplicate image_filename: {fn}")
        seen_files.add(fn)
        
        if rel_path in seen_paths:
            errors.append(f"Duplicate relative_image_path: {rel_path}")
        seen_paths.add(rel_path)
        
        # 2. Balanced classes
        if gt == 1:
            h_count += 1
        elif gt == 0:
            nh_count += 1
        else:
            errors.append(f"Invalid ground_truth for {sid}: {gt}")
            
        # 3. Stratum quota tracking
        if stratum in strata_counts:
            strata_counts[stratum] += 1
        else:
            errors.append(f"Unknown stratum {stratum} for {sid}")
            
        # 4. Image exists and readable
        p = image_dir / fn
        if not p.exists():
            errors.append(f"Image file does not exist: {p}")
        else:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Image {p} is corrupt: {e}")
                
        # 6. No historical evaluation meme
        if fn in QUARANTINED_HISTORICAL_MEMES:
            errors.append(f"Sample {fn} is in QUARANTINED_HISTORICAL_MEMES")
            
        # 7. No perceptual collision (d <= 3) with historical memes
        # 8. No OCR similarity collision (> 85%) with historical memes
        ocr = s.get("ocr_text", "")
        collision = check_candidate_against_quarantine(
            p, ocr, quarantined_data, max_hamming_distance=3, min_ocr_similarity=0.85
        )
        if collision:
            errors.append(f"Sample {fn} collides with historical {collision['quarantined_file']}: {collision}")
            
    if h_count != 50:
        errors.append(f"Humorous samples must be 50, got {h_count}")
    if nh_count != 50:
        errors.append(f"Non-humorous samples must be 50, got {nh_count}")
        
    for k, expected_q in STRATA_QUOTAS.items():
        actual_q = strata_counts.get(k, 0)
        if actual_q != expected_q["quota"]:
            errors.append(f"Stratum {k} count mismatch: expected {expected_q['quota']}, got {actual_q}")
            
    # Check internal pairwise collisions
    sample_hashes = {}
    for s in samples:
        fn = s["image_filename"]
        p = image_dir / fn
        sample_hashes[fn] = compute_average_hash(p)
        
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            fn1 = samples[i]["image_filename"]
            fn2 = samples[j]["image_filename"]
            d = hamming_distance(sample_hashes[fn1], sample_hashes[fn2])
            if d <= 3:
                errors.append(f"Internal perceptual hash collision between {fn1} and {fn2} (d={d})")
                
            ocr1 = samples[i].get("ocr_text", "")
            ocr2 = samples[j].get("ocr_text", "")
            if len(normalize_ocr_text(ocr1)) >= 15 and len(normalize_ocr_text(ocr2)) >= 15:
                sim = compute_ocr_similarity(ocr1, ocr2)
                if sim >= 0.85:
                    errors.append(f"Internal OCR collision between {fn1} and {fn2} (similarity={sim:.3f})")
                    
    return errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and validate held-out evaluation manifest.")
    parser.add_argument("--test-csv", default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--image-dir", default="data/processed/images", help="Path to image directory")
    parser.add_argument("--output", default="results/evaluation_heldout_100/sample_manifest.json", help="Primary manifest output path")
    parser.add_argument("--also-save", default="data/processed/sample_manifest.json", help="Secondary manifest output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    test_csv_path = Path(args.test_csv)
    image_dir = Path(args.image_dir)
    primary_out = Path(args.output)
    secondary_out = Path(args.also_save) if args.also_save else None
    
    print(f"Loading dataset from: {test_csv_path}")
    if not test_csv_path.exists():
        print(f"ERROR: Dataset file not found: {test_csv_path}")
        sys.exit(1)
        
    test_csv = pd.read_csv(test_csv_path)
    print(f"Total source records: {len(test_csv)}")
    
    print(f"Building reference data for 20 quarantined historical memes...")
    quarantined_data = build_quarantined_reference_data(test_csv, image_dir)
    print(f"Quarantined reference records: {len(quarantined_data)}")
    
    print("Filtering candidates and checking for collisions...")
    clean_candidates_df, exclusions = filter_candidate_pool(test_csv, image_dir, quarantined_data)
    print(f"Clean candidates remaining: {len(clean_candidates_df)}")
    print(f"  Clean humorous: {len(clean_candidates_df[clean_candidates_df['is_humorous'] == 1])}")
    print(f"  Clean non-humorous: {len(clean_candidates_df[clean_candidates_df['is_humorous'] == 0])}")
    print(f"Total exclusions: {len(exclusions)}")
    
    print(f"Sampling 100 memes with seed {args.seed} across 11 strata...")
    selected_samples = sample_heldout_manifest(clean_candidates_df, seed=args.seed)
    
    print("Building manifest document...")
    manifest_doc = build_manifest_document(
        selected_samples,
        image_dir,
        exclusions,
        clean_candidates_count=len(clean_candidates_df),
        total_source_count=len(test_csv),
        seed=args.seed
    )
    
    print("Performing exhaustive offline validation...")
    validation_errors = validate_manifest(manifest_doc, image_dir, quarantined_data)
    if validation_errors:
        print("VALIDATION FAILED with errors:")
        for err in validation_errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)
    print("Validation PASSED (0 errors).")
    
    # Save primary manifest
    primary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(primary_out, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)
    print(f"Saved primary manifest: {primary_out} ({primary_out.stat().st_size} bytes)")
    
    # Save secondary manifest if requested
    if secondary_out:
        secondary_out.parent.mkdir(parents=True, exist_ok=True)
        with open(secondary_out, "w", encoding="utf-8") as f:
            json.dump(manifest_doc, f, indent=2)
        print(f"Saved secondary manifest: {secondary_out} ({secondary_out.stat().st_size} bytes)")
        
    print("\nManifest Generation Completed Successfully.")


if __name__ == "__main__":
    main()
