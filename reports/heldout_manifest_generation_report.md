# Held-Out Evaluation Manifest Generation & Validation Report

**Document Type**: Dataset Engineering & Validation Report  
**Target Plan**: `reports/heldout_evaluation_plan.md`  
**Safety Protocol Reference**: `reports/heldout_evaluation_safety_fixes.md`  
**Generated Manifests**:  
- Primary Path: `results/evaluation_heldout_100/sample_manifest.json` (139,635 bytes)  
- Parity Path: `data/processed/sample_manifest.json` (139,635 bytes)  
**Author**: Antigravity Assistant  
**Date**: October 20, 2026  
**Status**: Generated, Deterministically Audited, and 100% Validated (63/63 Tests Passing)  

---

## Executive Summary

The 100-meme held-out evaluation dataset for the Drishtikon humour-detection project has been generated, stratified, deduplicated, and exhaustively validated offline. 

The dataset adheres strictly to the corrected sampling quotas from `reports/heldout_evaluation_plan.md` (rebalancing `NH3` from 10 to 4 and reallocating to `NH4=10`, `NH5=6`, `NH6=6`). All 20 historical evaluation memes from previous runs (`results/evaluation_twostage_v4/` and earlier) were quarantined, along with 15 perceptual average-hash collisions ($d \le 3$) discovered in the candidate pool. Zero OCR text collisions were detected.

The manifest was generated deterministically using a fixed random seed (`seed = 42`), leaving ground-truth labels completely intact. All 100 image files exist on disk, are fully readable via PIL, and possess zero duplicate paths or identifiers. Exactly 50 humorous and 50 non-humorous memes were selected across all 11 strata.

**Zero VLM inference was launched**, no model weights were loaded, and no evaluation runs were started.

---

## 1. Candidate Pool Audit & Exclusion Breakdown

### 1.1 Source Dataset
- **Source File**: `data/processed/test.csv` (Memotion 3 De-Factify 2 shared task, AAAI 2023).
- **Total Source Records**: 700 memes.
  - Initial Humorous ($y = 1$): 599 records (85.57%)
  - Initial Non-Humorous ($y = 0$): 101 records (14.43%)
- **Image Directory**: `data/processed/images` (all 700 image files verified present and valid).

### 1.2 Step 1: Historical Evaluation Quarantining ($N = 20$)
The 20 memes evaluated across historical runs (`results/evaluation_twostage_v4/`, `results/evaluation_1790*`, etc.) were quarantined to eliminate test leakage:

```python
QUARANTINED_HISTORICAL_MEMES = {
    "train_55.jpg", "train_127.jpg", "train_833.jpg", "train_1611.jpg",
    "train_2000.jpg", "train_2183.jpg", "train_2573.jpg", "train_2837.jpg",
    "train_3280.jpg", "train_3395.jpg", "train_4020.jpg", "train_4136.jpg",
    "train_4199.jpg", "train_4917.jpg", "train_5088.jpg", "train_5266.jpg",
    "train_5550.jpg", "train_5723.jpg", "train_5890.jpg", "train_6346.jpg"
}
```
- **Historical Exclusions**: 20 memes (10 humorous, 10 non-humorous).
- **Candidates Remaining**: 680 memes (589 humorous, 91 non-humorous).

### 1.3 Step 2: Perceptual Hash Deduplication ($d \le 3$)
Using the pure PIL/NumPy average hash implementation (`src/evaluation/deduplication.py`), all 680 candidate memes were compared against the 20 quarantined reference memes. 

A total of **15 perceptual collisions** ($d \le 3$) were identified and excluded:

| Candidate Filename | Ground Truth ($y$) | Quarantined Reference | Hamming Distance ($d$) | Collision Type | Notes / Content Match |
|:---|:---:|:---|:---:|:---|:---|
| `train_6993.jpg` | 1 | `train_2837.jpg` | **0** | Exact Perceptual Clone | Identical image re-encoded with alternate compression |
| `train_5575.jpg` | 1 | `train_4199.jpg` | **0** | Exact Perceptual Clone | Identical visual layout |
| `train_181.jpg` | 0 | `train_2837.jpg` | 1 | Near-Duplicate | Minor crop variation of reference template |
| `train_1165.jpg` | 1 | `train_2837.jpg` | 1 | Near-Duplicate | Slight border padding variation |
| `train_3747.jpg` | 1 | `train_5723.jpg` | 1 | Near-Duplicate | Near-identical visual structure |
| `train_627.jpg` | 1 | `train_5723.jpg` | 2 | Near-Duplicate | Resolution scaling variation |
| `train_3686.jpg` | 1 | `train_6346.jpg` | 2 | Near-Duplicate | Minor visual artifact variation |
| `train_4249.jpg` | 1 | `train_2000.jpg` | 3 | Template Variant | Same visual meme macro template |
| `train_5960.jpg` | 1 | `train_5723.jpg` | 3 | Template Variant | Same character pose template |
| `train_6626.jpg` | 1 | `train_2837.jpg` | 3 | Template Variant | Same two-panel layout |
| `train_3467.jpg` | 1 | `train_5723.jpg` | 3 | Template Variant | Similar background composition |
| `train_3056.jpg` | 1 | `train_5723.jpg` | 3 | Template Variant | Similar lighting and framing |
| `train_3980.jpg` | 0 | `train_5723.jpg` | 3 | Template Variant | Near-identical background |
| `train_3067.jpg` | 1 | `train_2837.jpg` | 3 | Template Variant | Common macro template |
| `train_703.jpg` | 1 | `train_2837.jpg` | 3 | Template Variant | Common macro template |

- **Perceptual Exclusions**: 15 memes (13 humorous, 2 non-humorous).

### 1.4 Step 3: Normalized OCR Similarity Check ($> 85\%$)
- SequenceMatcher Levenshtein ratio across normalized OCR text strings (min length $\ge 15$ chars).
- **OCR Collisions Detected**: **0** memes.

### 1.5 Clean Eligible Pool Summary
| Pool State | Humorous ($y=1$) | Non-Humorous ($y=0$) | Total Records |
|:---|:---:|:---:|:---:|
| Total Source (`test.csv`) | 599 | 101 | 700 |
| Excluded: Historical Quarantined | -10 | -10 | -20 |
| Excluded: Perceptual aHash Collisions | -13 | -2 | -15 |
| Excluded: Normalized OCR Collisions | 0 | 0 | 0 |
| Excluded: Missing or Corrupt Files | 0 | 0 | 0 |
| **Final Eligible Candidate Pool** | **576** | **89** | **665** |

Both humorous (576 eligible vs. 50 quota) and non-humorous (89 eligible vs. 50 quota) pools comfortably exceed the required sample targets.

---

## 2. Stratification Taxonomy & Final Sample Allocation ($N = 100$)

All 100 memes were sampled deterministically using `random_seed = 42`. The selection strictly fills the planned quota across all 11 strata without moving samples between classes or altering ground-truth labels.

```
Final Held-Out Benchmark: N = 100
├── Humorous Class (is_humorous = 1): N = 50
│     ├── H1: High Cultural Dependency (Bollywood, Cricket, Festivals, Family)     [Allocated: 15 / Quota: 15]
│     ├── H2: Hinglish Daily Life & Situational Irony (Code-switching, Tropes)     [Allocated: 15 / Quota: 15]
│     ├── H3: Political Satire & Societal Commentary (Twisted meaning, Irony)      [Allocated: 10 / Quota: 10]
│     ├── H4: Dark Humour & Edgy Comedy (Offensive/taboo comedic mechanisms)       [Allocated:  5 / Quota:  5]
│     └── H5: Universal / Non-Cultural Situational Comedy (Slapstick, Animals)     [Allocated:  5 / Quota:  5]
│
└── Non-Humorous Class (is_humorous = 0): N = 50
      ├── NH1: Serious News, Tragedy, & Current Affairs (COVID, Public Order)      [Allocated: 12 / Quota: 12]
      ├── NH2: Motivational Quotes & Inspirational Sayings (Philosophical Quotes)  [Allocated: 12 / Quota: 12]
      ├── NH3: Factual, Educational, & Historical Graphics (Infographics, Stats)   [Allocated:  4 / Quota:  4]
      ├── NH4: Sarcastic Non-Humour (Subtle/Cynical statements labelled 0)         [Allocated: 10 / Quota: 10]
      ├── NH5: Hostile / Aggressive Social Media Statements (Polarized attacks)    [Allocated:  6 / Quota:  6]
      └── NH6: Commercial Advertisements & Watermarked Spam (Product flyers)       [Allocated:  6 / Quota:  6]
```

### 2.1 Complete Stratum Allocation Table

| Stratum ID | Class | Stratum Name | Eligible Pool | Sampled Quota | Allocation Status | Primary Failure Mode Tested |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **H1** | Humorous | High Cultural Dependency | 292 | **15** | 100% Filled | Stage-2 cultural retrieval necessity |
| **H2** | Humorous | Hinglish Daily Life & Banter | 179 | **15** | 100% Filled | Romanized Hindi phonetic understanding |
| **H3** | Humorous | Political Satire & Commentary | 475 | **10** | 100% Filled | Subtle rhetorical irony vs literal text |
| **H4** | Humorous | Dark Humour & Edgy Comedy | 221 | **5** | 100% Filled | Model conflating taboo with non-humour |
| **H5** | Humorous | Universal / Non-Cultural | 198 | **5** | 100% Filled | Stage-1 gating clean short-circuit |
| **NH1** | Non-Humorous | Serious News & Tragedy | 19 | **12** | 100% Filled | Anti-absurdity: tragic news hallucination |
| **NH2** | Non-Humorous | Motivational Quotes | 23 | **12** | 100% Filled | Moral/philosophical text recognition |
| **NH3** | Non-Humorous | Factual & Educational Graphics | 4 | **4** | 100% Filled | Non-comedic informational graphics |
| **NH4** | Non-Humorous | Sarcastic Non-Humour | 56 | **10** | 100% Filled | Cynicism vs true comedy boundary |
| **NH5** | Non-Humorous | Hostile / Aggressive Statements | 22 | **6** | 100% Filled | Provocation without comedic punchline |
| **NH6** | Non-Humorous | Commercial Advertisements & Spam | 6 | **6** | 100% Filled | Spam rejection and watermark noise |
| **Total** | — | **Full Held-Out Evaluation Set** | **665** | **100** | **100% Filled** | **Balanced 50 Humorous / 50 Non-Humorous** |

---

## 3. Human Review & Ambiguity Metadata Protocol

In compliance with §5.3 of `reports/heldout_evaluation_plan.md`, **no ground-truth label was altered**. Instead, each candidate received an immutable metadata flag (`ambiguity_status` and `ambiguity_reason`) to enable dual-metric reporting:
1. **Full Benchmark Set ($N = 100$)**: Unedited benchmark against official Memotion 3 ground-truth.
2. **Clean Consensus Subset ($N = 78$)**: Disaggregated subset excluding known borderline ambiguous samples.

### Breakdown of Ambiguity Classifications
- **`CLEAN_CONSENSUS`**: **78 samples** (78.0%)
- **`AMBIGUOUS_BORDERLINE`**: **22 samples** (22.0%)
  - *Tragic news or health advisory formatted in meme card template* (`NH1`): 10 samples (e.g. `train_5634.jpg`, `train_5608.jpg`, `train_3949.jpg`).
  - *Biting cynicism or dark sarcasm labelled not_funny by crowd annotators* (`NH4`): 9 samples (e.g. `train_5883.jpg`, `train_3675.jpg`, `train_6614.jpg`).
  - *Edgy or taboo comedy with potential moral disapproval* (`H4`): 3 samples (e.g. `train_6155.jpg`, `train_1316.jpg`).

---

## 4. Reproducibility & Determinism Controls

- **Random Generator**: Python `random.Random(42)`.
- **Baseline Candidate Ordering**: Candidates sorted deterministically by `image_filename` before sampling.
- **Priority-Ordered Sampling**: Rare strata (`NH3`, `NH6`, `H5`, `H4`) sampled first to guarantee disjoint sets without candidate starvation.
- **Verification of Determinism**: Re-sampling from `data/processed/test.csv` with `seed = 42` produces a bit-for-bit identical sample list (validated in `tests/test_heldout_manifest.py`).

---

## 5. Offline Validation & Test Suite Results

The manifest was validated through both internal generator assertions and a dedicated offline test suite (`tests/test_heldout_manifest.py`).

### 5.1 Test Execution Matrix

```bash
python -m pytest tests/test_checkpoint.py tests/test_prefill_timeout.py tests/test_drishtikon_lexical_mitigation.py tests/test_evaluation_safety.py tests/test_heldout_manifest.py
```

| Test Suite File | Scope | Tests Run | Passed | Failed | Key Verification Point |
|:---|:---|:---:|:---:|:---:|:---|
| `tests/test_heldout_manifest.py` | `TestHeldoutManifestIntegrity` | 11 | 11 | 0 | Exactly 100 unique samples, 50/50 balance, all strata quotas fulfilled, image readability, zero historical/perceptual/OCR collisions, runner schema compatibility, dual-path parity, reproducibility, and actionable failure handling. |
| `tests/test_evaluation_safety.py` | Safety & Failure Handling | 11 | 11 | 0 | Timeout records `prediction=None`, JSON failures handled, denominators exclude failures, resume safety, config isolation. |
| `tests/test_checkpoint.py` | Checkpoint Engine | 30 | 30 | 0 | Atomic writing, JSONL appends, status reporting, stop signals. |
| `tests/test_prefill_timeout.py` | Prefill Watchdog | 4 | 4 | 0 | Slow prefill does not trigger token 1 timeout; decoding limit enforced. |
| `tests/test_drishtikon_lexical_mitigation.py` | Lexical Mitigation | 7 | 7 | 0 | Section 144/302/377/420 negative controls pass; Metro positive control intact. |
| **Total** | **Full Project Test Suite** | **63** | **63** | **0** | **100% Pass Rate** |

---

## 6. Manifest File Schema & Compatibility

The manifest is saved as a single immutable JSON document compatible with `scripts/run_controlled_evaluation.py`:

```json
{
  "manifest_version": "1.0.0",
  "created_at": "2026-10-20",
  "random_seed": 42,
  "sample_size": 100,
  "target_balance": { "humorous": 50, "non_humorous": 50 },
  "ambiguity_breakdown": {
    "clean_consensus_samples": 78,
    "ambiguous_borderline_samples": 22
  },
  "strata_counts": {
    "NH3": 4, "NH6": 6, "NH5": 6, "NH1": 12, "NH2": 12, "NH4": 10,
    "H5": 5, "H4": 5, "H3": 10, "H1": 15, "H2": 15
  },
  "samples": [
    {
      "sample_id": "heldout_001",
      "sample_index": 0,
      "image_filename": "train_5353.jpg",
      "relative_image_path": "data/processed/images/train_5353.jpg",
      "ground_truth": 1,
      "is_humorous": 1,
      "class_name": "humorous",
      "stratum_id": "H3",
      "stratum_name": "Political Satire & Societal Commentary",
      "stratum_description": "Subtle rhetorical irony, governance, public affairs",
      "source_dataset": "Memotion 3 (De-Factify 2, AAAI 2023)",
      "source_split": "data/processed/test.csv",
      "original_labels": {
        "humour": "very_funny",
        "sarcastic": "twisted_meaning",
        "offensive": "not_offensive",
        "motivational": "not_motivational",
        "overall": "positive"
      },
      "ocr_text": "GOA TO ME EVERY YEAR:  Hai himmat?.. Aayega?.. Aayega?",
      "file_md5": "4508ecf661bb22fec92c011e4f1a2380",
      "ahash_bitstring": "0000000000000000000111000011111000111110001111100011111000011000",
      "deduplication_status": {
        "quarantine_overlap": false,
        "perceptual_ahash_collision": false,
        "ocr_similarity_collision": false,
        "verified": true
      },
      "ambiguity_status": "CLEAN_CONSENSUS",
      "ambiguity_reason": null
    }
  ]
}
```

---

## 7. Inventory of Modified and Created Files

### Files Created
1. `scripts/create_heldout_manifest.py`: Deterministic manifest generator and offline validator.
2. `tests/test_heldout_manifest.py`: Offline test suite containing 11 tests verifying all manifest integrity criteria.
3. `results/evaluation_heldout_100/sample_manifest.json`: Primary immutable held-out evaluation manifest (139,635 bytes).
4. `data/processed/sample_manifest.json`: Secondary parity copy of manifest (139,635 bytes).
5. `reports/heldout_manifest_generation_report.md`: This comprehensive generation and validation report.

### Files Preserved Untouched
- Model weights, checkpoints, prompts, and classification thresholds.
- Official ground-truth labels in `data/processed/test.csv`.
- All historical evaluation outputs in `results/evaluation_twostage_v4/` and `reports/v4_*.md`.

---

## 8. Audit Statement & Model Safety Confirmation

- **VLM Inferences Executed**: **0** (Zero Qwen forward passes or embeddings were run).
- **Model Loading**: No model weights, torch neural networks, or evaluation servers were initialized.
- **Manifest Status**: Frozen, fully validated, and ready for chunked execution across Configurations A, B, and C upon user authorization.
