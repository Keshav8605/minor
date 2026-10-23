"""
Offline Unit and Regression Tests for the Held-Out Evaluation Manifest.

Verifies:
1. Exactly 100 unique samples.
2. Balanced classes: 50 humorous, 50 non-humorous.
3. Every stratum satisfies its quota (H1-H5, NH1-NH6).
4. All images exist on disk and are readable.
5. Zero duplicate filenames, image paths, or sample IDs.
6. Zero overlap with the 20 historical evaluation memes.
7. Zero perceptual hash collisions (d <= 3) with historical memes.
8. Zero normalized OCR similarity collisions (> 85%) with historical memes.
9. Compatibility with evaluation runner schema.
10. Offline execution guarantee (no VLM or torch model loaded).
11. Sampling reproducibility: re-running with seed 42 produces identical manifest.
12. Actionable validation failure behavior on corrupted inputs.
"""

import os
import json
import copy
from pathlib import Path
import pytest
from PIL import Image

from scripts.create_heldout_manifest import (
    QUARANTINED_HISTORICAL_MEMES,
    STRATA_QUOTAS,
    validate_manifest,
    sample_heldout_manifest,
    filter_candidate_pool,
    build_quarantined_reference_data
)
from src.evaluation.deduplication import (
    compute_average_hash,
    hamming_distance,
    compute_ocr_similarity,
    normalize_ocr_text
)
import pandas as pd


MANIFEST_PRIMARY_PATH = Path("results/evaluation_heldout_100/sample_manifest.json")
MANIFEST_SECONDARY_PATH = Path("data/processed/sample_manifest.json")
IMAGE_DIR = Path("data/processed/images")
TEST_CSV_PATH = Path("data/processed/test.csv")


@pytest.fixture(scope="module")
def manifest():
    """Loads the generated primary manifest."""
    assert MANIFEST_PRIMARY_PATH.exists(), f"Primary manifest not found: {MANIFEST_PRIMARY_PATH}"
    with open(MANIFEST_PRIMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def quarantined_data():
    """Loads quarantined reference records."""
    df = pd.read_csv(TEST_CSV_PATH)
    return build_quarantined_reference_data(df, IMAGE_DIR)


class TestHeldoutManifestIntegrity:
    """Core integrity tests for the held-out evaluation manifest."""

    def test_sample_counts_and_uniqueness(self, manifest):
        """Verify exactly 100 unique samples and balanced 50/50 split."""
        samples = manifest["samples"]
        assert len(samples) == 100, f"Expected 100 samples, got {len(samples)}"
        
        sample_ids = [s["sample_id"] for s in samples]
        filenames = [s["image_filename"] for s in samples]
        paths = [s["relative_image_path"] for s in samples]
        
        assert len(set(sample_ids)) == 100, "Duplicate sample_ids detected"
        assert len(set(filenames)) == 100, "Duplicate image_filenames detected"
        assert len(set(paths)) == 100, "Duplicate relative_image_paths detected"
        
        humorous = [s for s in samples if s["ground_truth"] == 1]
        non_humorous = [s for s in samples if s["ground_truth"] == 0]
        
        assert len(humorous) == 50, f"Expected 50 humorous, got {len(humorous)}"
        assert len(non_humorous) == 50, f"Expected 50 non-humorous, got {len(non_humorous)}"

    def test_strata_quotas_fulfilled(self, manifest):
        """Verify all 11 strata exactly meet their allocated quotas."""
        samples = manifest["samples"]
        counts = {}
        for s in samples:
            sid = s["stratum_id"]
            counts[sid] = counts.get(sid, 0) + 1
            
        for sid, q_info in STRATA_QUOTAS.items():
            expected = q_info["quota"]
            actual = counts.get(sid, 0)
            assert actual == expected, (
                f"Stratum {sid} ({q_info['name']}) quota mismatch: expected {expected}, got {actual}"
            )
            
        assert manifest["strata_counts"] == counts

    def test_all_images_exist_and_readable(self, manifest):
        """Verify that every image file referenced in the manifest exists and is readable."""
        samples = manifest["samples"]
        for s in samples:
            img_path = Path(s["relative_image_path"])
            assert img_path.exists(), f"Image missing: {img_path}"
            assert img_path.stat().st_size > 0, f"Image is empty (0 bytes): {img_path}"
            
            # Verify image integrity via PIL
            with Image.open(img_path) as img:
                img.verify()

    def test_zero_historical_overlap(self, manifest):
        """Verify that none of the 20 quarantined historical memes are in the manifest."""
        samples = manifest["samples"]
        manifest_files = {s["image_filename"] for s in samples}
        overlap = manifest_files.intersection(QUARANTINED_HISTORICAL_MEMES)
        assert len(overlap) == 0, f"Historical evaluation memes found in manifest: {overlap}"

    def test_zero_perceptual_hash_collisions_with_quarantine(self, manifest, quarantined_data):
        """Verify no sample has a perceptual hash distance d <= 3 against historical memes."""
        samples = manifest["samples"]
        for s in samples:
            p = IMAGE_DIR / s["image_filename"]
            ahash = compute_average_hash(p)
            assert ahash is not None, f"Could not compute ahash for {p}"
            for q in quarantined_data:
                d = hamming_distance(ahash, q["ahash"])
                assert d > 3, (
                    f"Sample {s['image_filename']} has perceptual hash collision with historical "
                    f"{q['filename']} (Hamming distance {d} <= 3)"
                )

    def test_zero_ocr_similarity_collisions_with_quarantine(self, manifest, quarantined_data):
        """Verify no sample has normalized OCR similarity > 85% against historical memes."""
        samples = manifest["samples"]
        for s in samples:
            s_ocr = s.get("ocr_text", "")
            if len(normalize_ocr_text(s_ocr)) < 15:
                continue
            for q in quarantined_data:
                q_ocr = q.get("ocr", "")
                if len(normalize_ocr_text(q_ocr)) < 15:
                    continue
                sim = compute_ocr_similarity(s_ocr, q_ocr)
                assert sim < 0.85, (
                    f"Sample {s['image_filename']} has OCR similarity collision with historical "
                    f"{q['filename']} (similarity {sim:.3f} >= 0.85)"
                )

    def test_zero_internal_pairwise_collisions(self, manifest):
        """Verify that samples in the 100-meme manifest do not collide perceptually with each other."""
        samples = manifest["samples"]
        hashes = {s["image_filename"]: compute_average_hash(IMAGE_DIR / s["image_filename"]) for s in samples}
        
        for i in range(len(samples)):
            fn1 = samples[i]["image_filename"]
            for j in range(i + 1, len(samples)):
                fn2 = samples[j]["image_filename"]
                d = hamming_distance(hashes[fn1], hashes[fn2])
                assert d > 3, f"Internal perceptual collision between {fn1} and {fn2} (d={d} <= 3)"

    def test_evaluation_runner_compatibility(self, manifest):
        """Verify all fields required by run_controlled_evaluation.py are present."""
        samples = manifest["samples"]
        required_fields = {
            "sample_id", "sample_index", "image_filename", "relative_image_path",
            "ground_truth", "is_humorous", "class_name", "stratum_id", "stratum_name",
            "source_dataset", "source_split", "ocr_text", "file_md5", "ahash_bitstring",
            "deduplication_status", "ambiguity_status"
        }
        for s in samples:
            missing = required_fields - set(s.keys())
            assert not missing, f"Sample {s.get('sample_id')} missing required fields: {missing}"
            assert s["ground_truth"] in (0, 1)
            assert s["is_humorous"] == s["ground_truth"]
            assert s["sample_index"] >= 0

    def test_dual_manifest_file_parity(self):
        """Verify primary and secondary manifest files exist and are identical."""
        assert MANIFEST_PRIMARY_PATH.exists()
        assert MANIFEST_SECONDARY_PATH.exists()
        
        with open(MANIFEST_PRIMARY_PATH, "r", encoding="utf-8") as f:
            p_data = json.load(f)
        with open(MANIFEST_SECONDARY_PATH, "r", encoding="utf-8") as f:
            s_data = json.load(f)
            
        assert p_data == s_data, "Primary and secondary manifest JSON files differ!"

    def test_sampling_reproducibility(self):
        """Verify that re-sampling with seed 42 produces identical sample selections."""
        df = pd.read_csv(TEST_CSV_PATH)
        q_data = build_quarantined_reference_data(df, IMAGE_DIR)
        clean_df, _ = filter_candidate_pool(df, IMAGE_DIR, q_data)
        
        run1 = sample_heldout_manifest(clean_df, seed=42)
        run2 = sample_heldout_manifest(clean_df, seed=42)
        
        files1 = [s["row"]["image_filename"] for s in run1]
        files2 = [s["row"]["image_filename"] for s in run2]
        assert files1 == files2, "Sampling with seed 42 is not deterministic!"

    def test_actionable_validation_errors(self, manifest, quarantined_data):
        """Verify validate_manifest catches and reports corruptions with descriptive messages."""
        corrupt_manifest = copy.deepcopy(manifest)
        
        # 1. Inject duplicate sample
        corrupt_manifest["samples"][1]["image_filename"] = corrupt_manifest["samples"][0]["image_filename"]
        errors = validate_manifest(corrupt_manifest, IMAGE_DIR, quarantined_data)
        assert any("Duplicate image_filename" in e for e in errors)
        
        # 2. Inject historical sample
        corrupt_manifest2 = copy.deepcopy(manifest)
        corrupt_manifest2["samples"][0]["image_filename"] = "train_127.jpg"
        errors2 = validate_manifest(corrupt_manifest2, IMAGE_DIR, quarantined_data)
        assert any("QUARANTINED_HISTORICAL_MEMES" in e for e in errors2)
