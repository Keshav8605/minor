# Comprehensive Held-Out Evaluation Plan: Drishtikon Meme Humour Detection (N=100)

**Document Type**: Experimental Design & Evaluation Specification  
**Project**: Drishtikon Multi-Modal Hindi/Hinglish Meme Humour Detection  
**Target Sample Size**: $N = 100$ Memes (50 Humorous, 50 Non-Humorous)  
**Execution Mode**: Offline, Deterministic, Zero-Inference Planning (Read-Only)  
**Author**: Antigravity Assistant  
**Date**: October 20, 2026  
**Status**: Ready for Execution Upon User Authorization  

---

## 1. Executive Summary & Objectives

Past evaluations of the Drishtikon two-stage meme humour detection pipeline (evaluations v1 through v4) have been constrained to small exploratory batches ($N = 10\text{--}20$ memes, with v4 focusing on 11 memes). While these runs successfully uncovered key system mechanics—such as the 8/11 short-circuit gating behavior, the single-token DRISHTIKON lexical collision, and probability calibration dynamics—they were insufficient to establish statistically robust performance benchmarks or validate optimal decision thresholds.

This plan specifies the methodology, data selection, stratification, execution protocol, and resource estimates for a **statistically rigorous, held-out evaluation of 100 memes**.

### Core Objectives
1. **Class Balance**: Overcome the heavy natural class skew of the Memotion 3 dataset (~85% humorous) by establishing an exact 50:50 balance ($N_{\text{humour}}=50, N_{\text{non-humour}}=50$).
2. **Representative Multi-Dimensional Stratification**: Span cultural dependency, linguistic code-switching (Hinglish/Hindi/English), comedic mechanisms (satire, dark humour, situational irony), and critical negative subgenres (serious news juxtaposition, motivational quotes, commercial spam).
3. **Strict Data Leakage Prevention**: Quarantine all 20 historical evaluation memes and ensure complete independence between threshold-tuning data and final held-out test data.
4. **Label Provenance & Ambiguity Management**: Preserve ground-truth labels non-destructively while flagging ambiguous crowdsourced annotations for disaggregated analysis.
5. **Benchmarking Protocol**: Compare the **Frozen General Baseline**, the **Frozen Cultural-Aware Baseline (with Section 144 mitigation)**, and a proposed **Targeted Prompt Refinement**.
6. **Operational Feasibility**: Account for observed CPU inference costs (~24 hours total) and establish atomic checkpointing to guarantee execution safety.

---

## 2. Dataset Inventory & Source Material Selection

### 2.1 Available Labelled Data

The repository contains processed Memotion 3 splits in `data/processed/` along with 8,500 corresponding images in `data/processed/images/`:

| Dataset File | Total Rows | Humorous ($1$) | Non-Humorous ($0$) | Natural Class Skew | Image Availability | Overlap with Past Evals |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `data/processed/train.csv` | 5,600 | 4,792 (85.6%) | 808 (14.4%) | 5.93 : 1 | 5,600 images on disk | **0 memes** |
| `data/processed/validation.csv` | 700 | 599 (85.6%) | 101 (14.4%) | 5.93 : 1 | 700 images on disk | **0 memes** |
| `data/processed/test.csv` | 700 | 599 (85.6%) | 101 (14.4%) | 5.93 : 1 | 700 images on disk | **20 memes** (historical evaluations) |
| `data/processed/blind_test.csv` | 1,500 | Unlabelled | Unlabelled | N/A | 1,500 images on disk | None |

### 2.2 Split Allocation & Separation Strategy

To ensure strict experimental integrity:
- **Development & Threshold Tuning Pool (`data/processed/validation.csv`, $N=700$)**:
  Reserved exclusively for exploratory sweeps, Stage-1 gating sensitivity calibration, and retriever hyperparameter tuning. Zero memes from this pool may enter the final held-out test set.
- **Held-Out Evaluation Pool (`data/processed/test.csv`, $N=680$ clean candidates)**:
  All past evaluations (v1 through v4) sampled from `test.csv`. After permanently excluding the 20 historical evaluation memes, exactly **680 pristine, unexamined memes** remain in `test.csv` (91 non-humorous, 589 humorous).
- **Source Selection**:
  The 100-meme held-out evaluation set (`test_heldout_100`) will be constructed exclusively from the 680 unused candidates in `data/processed/test.csv`. This adheres strictly to standard machine learning protocol by keeping the test split segregated from training and validation splits.

---

## 3. Strict Quarantine & Overlap Prevention

### 3.1 Historical Evaluation Quarantine Blacklist

In prior evaluation runs and error analyses (`results/evaluation_1790249336/`, `results/evaluation_twostage_v4/`, `reports/v4_error_analysis.md`, `reports/v4_threshold_sweep_verified.md`), exactly 20 unique memes were evaluated:

```python
QUARANTINED_HISTORICAL_MEMES = {
    "train_55.jpg", "train_127.jpg", "train_833.jpg", "train_1611.jpg",
    "train_2000.jpg", "train_2183.jpg", "train_2573.jpg", "train_2837.jpg",
    "train_3280.jpg", "train_3395.jpg", "train_4020.jpg", "train_4136.jpg",
    "train_4199.jpg", "train_4917.jpg", "train_5088.jpg", "train_5266.jpg",
    "train_5550.jpg", "train_5723.jpg", "train_5890.jpg", "train_6346.jpg"
}
```

### 3.2 Overlap Prevention Guarantees
1. **Filename Exclusion**: An automated assertion in the sampling script will check `assert set(sampled_filenames).isdisjoint(QUARANTINED_HISTORICAL_MEMES)`.
2. **Perceptual Image Hash Deduplication**: Using `imagehash.average_hash`, all candidate images will be checked against the 20 quarantined images with a Hamming distance threshold of $d \le 3$ to eliminate identical or re-encoded image duplicates.
3. **OCR Text Normalization Matching**: Normalized text matching ($> 85\%$ Levenshtein similarity) will be enforced to prevent near-duplicate meme captions from entering the test set.

---

## 4. Stratification Taxonomy & Subgroup Allocation ($N=100$)

To create a balanced, representative evaluation set, the 100 memes will be partitioned equally into **50 Humorous** and **50 Non-Humorous** samples across four orthogonal dimensions.

```
Total Held-Out Evaluation Set: N = 100
├── Humorous Class (is_humorous = 1): N = 50
│     ├── Stratum H1: High Cultural Dependency (Bollywood, Cricket, Festivals, Family) [N = 15]
│     ├── Stratum H2: Hinglish Daily Life & Situational Irony [N = 15]
│     ├── Stratum H3: Political Satire & Societal Commentary [N = 10]
│     ├── Stratum H4: Dark Humour & Edgy Comedy [N = 5]
│     └── Stratum H5: Universal / Non-Cultural Situational Comedy [N = 5]
│
└── Non-Humorous Class (is_humorous = 0): N = 50
      ├── Stratum NH1: Serious News, Tragedy, & Current Affairs Juxtaposition [N = 12]
      ├── Stratum NH2: Motivational Quotes & Inspirational Sayings [N = 12]
      ├── Stratum NH3: Factual, Educational, & Historical Graphics [N = 4]
      ├── Stratum NH4: Sarcastic/Ironic Commentary Labelled Non-Humorous [N = 10]
      ├── Stratum NH5: Hostile / Aggressive Social Media Statements [N = 6]
      └── Stratum NH6: Commercial Advertisements & Watermarked Spam [N = 6]
```

### 4.1 Humorous Class Stratification Breakdown ($N=50$)

| Stratum ID | Stratum Name | Sampling Criteria & Data Filters | Sample Count | Primary Failure Mode Tested |
|:---|:---|:---|:---:|:---|
| **H1** | **High Cultural Dependency** | OCR matches `CATEGORY_KEYWORDS` (Bollywood films, cricket/IPL, arranged marriage, Indian joint family, puja). | **15** | Tests whether Stage-2 cultural retrieval provides necessary grounded context for Qwen. |
| **H2** | **Hinglish Daily Life & Banter** | Latin-script Hindi code-switching (`bhai`, `yaar`, `padhai`, `jugaad`, `padosi`), relatable middle-class tropes. | **15** | Tests VLM's phonetic understanding of informal Hinglish slang without formal grammar. |
| **H3** | **Political Satire & Commentary** | Memotion 3 `sarcastic in ['general', 'twisted_meaning']`, political figures, governance memes. | **10** | Evaluates detection of subtle rhetorical irony vs. literal message. |
| **H4** | **Dark Humour & Edgy Comedy** | Memotion 3 `offensive in ['slight', 'very_offensive']`, taboo or cynical comedic mechanisms. | **5** | Evaluates whether model conflates offensive tone with lack of humour. |
| **H5** | **Universal / General Humour** | Visual slapstick, animal memes, universal workplace tropes without Indian cultural entities. | **5** | Confirms that Stage-1 gating cleanly short-circuits without unnecessary cultural lookups. |

### 4.2 Non-Humorous Negative Class Stratification Breakdown ($N=50$)

The negative class is deliberately designed to challenge the model's known vulnerabilities identified in `reports/v4_error_analysis.md`:

| Stratum ID | Stratum Name | Sampling Criteria & Data Filters | Sample Count | Primary Failure Mode Tested |
|:---|:---|:---|:---:|:---|
| **NH1** | **Serious News & Tragic Juxtaposition** | Real news events, COVID updates, public order, crime reports formatted in meme-like image cards. | **12** | **Critical Anti-Absurdity Test**: Models frequently hallucinate humour in tragic news due to unexpected image-text pairings (e.g. `train_3280`, `train_1611`). |
| **NH2** | **Motivational & Inspirational Quotes** | Memotion 3 `motivational == 'motivational'`, portrait of historical figure with inspirational quote. | **12** | Tests model's ability to recognize serious moral or motivational intent without comedy. |
| **NH3** | **Factual & Educational Graphics** | Infographics, historical timelines, civic announcements, science facts. | **4** | Verifies rejection of non-comedic informational text cards (adjusted to match available Memotion 3 pool). |
| **NH4** | **Sarcastic Non-Humour (Subtle/Cynical)** | Memotion 3 `sarcastic in ['twisted_meaning', 'very_twisted']` but labelled `is_humorous == 0`. | **10** | Tests fine boundary between bitter cynicism and true comedy. |
| **NH5** | **Hostile / Aggressive Statements** | Polarized political or personal hostility without wit, parody, or punchline. | **6** | Verifies that provocative or shocking statements are not mistaken for jokes. |
| **NH6** | **Commercial Spam & Watermarked Ads** | Promotional product flyers, adult content links, social media handles with spam text (e.g. `train_5266`). | **6** | Verifies rejection of spam and noise. |

---

## 5. Label Provenance, Noise Vectors, & Human Review Protocol

### 5.1 Dataset Provenance & Label Generation History
- **Source Benchmark**: Memotion 3 dataset (De-Factify 2 shared task, AAAI 2023 / SemEval).
- **Collection Method**: Crowdsourced multi-annotator labelling across 4 distinct dimensions:
  - `humour`: 4-point Likert scale (`not_funny`, `funny`, `very_funny`, `hilarious`).
  - `is_humorous`: Deterministically binarized during pipeline preprocessing:
    $$\text{is\_humorous} = \begin{cases} 1, & \text{if humour} \in \{\text{'funny'}, \text{'very\_funny'}, \text{'hilarious'}\} \\ 0, & \text{if humour} == \text{'not\_funny'} \end{cases}$$
  - `sarcastic`: 4-point categorical (`not_sarcastic`, `general`, `twisted_meaning`, `very_twisted`).
  - `offensive`: 4-point scale (`not_offensive`, `slight`, `very_offensive`, `hateful_offensive`).
  - `motivational`: Binary (`not_motivational`, `motivational`).

### 5.2 Documented Crowdsourcing Noise Vectors
From `reports/v4_error_analysis.md`, three systematic label noise vectors must be acknowledged:
1. **Subtle Irony Blindness**: Annotators frequently label dry, deadpan, or highly localized Hinglish sarcasm as `not_funny` (0), creating false positives when models recognize the genuine satirical device.
2. **Tragic/Offensive Penalization**: Annotators often mark memes about sensitive political controversies or deaths as `not_funny` out of moral distaste, even when a comedic or parody template was employed.
3. **Subjective Amusement vs. Rhetorical Humour**: A meme may be crafted as a joke, but if the crowdsourced rater personally did not laugh, they mark `not_funny`.

### 5.3 Non-Destructive Ambiguity Flagging Protocol
To maintain strict scientific integrity, **no ground-truth label will be altered automatically or manually before evaluation**. Instead, each candidate in the 100-meme set will be audited by a human reviewer and assigned an immutable metadata flag:

- **`ambiguity_status`**: `CLEAN_CONSENSUS` vs. `AMBIGUOUS_BORDERLINE`.
- **`ambiguity_reason`**: Documenting the specific source of tension (e.g., *Tragic news template*, *Biting sarcasm vs cynicism*, *Subjective punchline*).
- **Dual Metric Reporting**:
  The evaluation harness will compute and report metrics on:
  1. **Full Benchmark Set ($N=100$)**: The authoritative unedited benchmark.
  2. **Clean Consensus Subset ($N \approx 75\text{--}80$)**: The disaggregated subset excluding borderline ambiguous samples, isolating pure algorithmic performance from crowdsourcing noise.

---

## 6. Experimental Comparison Protocol

The 100-meme held-out evaluation will benchmark three distinct configurations in a standardized, reproducible pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     100-Meme Held-Out Test Set                         │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
│ Configuration A: │       │ Configuration B: │        │ Configuration C: │
│ Frozen General   │       │ Frozen Cultural  │        │ Targeted Prompt  │
│ Baseline         │       │ Pipeline (v4)    │        │ Refinement       │
│                  │       │                  │        │                  │
│ • Stage 1 only   │       │ • Stage 1 Gating │        │ • Anti-Absurdity │
│ • No cultural RAG│       │ • DRISHTIKON RAG │        │   News Gating    │
│ • Temp = 0.0     │       │   (Mitigated)    │        │ • Hinglish Normal│
│                  │       │ • Stage 2 Qwen   │        │ • Calibrated Temp│
└──────────────────┘       └──────────────────┘        └──────────────────┘
```

### 6.1 Configuration Specifications

1. **Configuration A: Frozen General Baseline**
   - **Architecture**: Single-stage VLM inference.
   - **Prompt**: Standard general humour analysis prompt (`VLMInferenceEngine.GENERAL_PROMPT`).
   - **Cultural Input**: Disabled (`cultural_sources = []`).
   - **Threshold**: Default decision threshold $\tau = 0.50$ (and evaluated across full sweep $0.40\text{--}0.70$).
   - **Purpose**: Establishes the pure visual-linguistic capability of Qwen2.5-VL-7B without external retrieval.

2. **Configuration B: Frozen Cultural-Aware Baseline (v4 + Mitigation)**
   - **Architecture**: Two-stage adaptive gating (`CulturalContextBuilder`).
   - **Cultural Knowledge**: Integrated `PROJECT_JSON` + DRISHTIKON RAG with the **newly verified Section 144 content-token overlap mitigation** (`min_similarity_threshold = 0.25`).
   - **Gating**: Stage-1 output routes to Stage 2 only if `cultural_dependency in ['medium', 'high']`.
   - **Purpose**: Measures the empirical benefit of external cultural context retrieval and establishes the official benchmark for the current production architecture.

3. **Configuration C: Targeted Prompt Refinement (Future Candidate)**
   - **Architecture**: Two-stage gating with prompt enhancements targeting v4 error failure modes:
     - *Anti-Absurdity News Filter*: Explicit instruction preventing Qwen from classifying serious news as humour purely because of surprising image juxtaposition.
     - *Hinglish Phonetic Normalization*: Instruction directing the VLM to phonetically interpret non-standard Romanized Hindi spelling.
   - **Retriever**: Unchanged (uses mitigated DRISHTIKON).
   - **Purpose**: Evaluates whether prompt-level cognitive steering resolves the remaining false positives identified in `reports/v4_error_analysis.md`.

### 6.2 Determinism & Reproducibility Controls
- **Random Seed**: Fixed at `seed = 42` across all Python, PyTorch, and CUDA random number generators.
- **Decoding**: Greedy decoding (`do_sample = False`, `temperature = 0.0`, `top_p = 1.0`).
- **Input Text Consistency**: Pre-extracted OCR text from `test.csv` passed identically to all models.
- **Image Preprocessing**: Pre-scaled with fixed aspect ratio preserved, capped at max resolution (1024x1024) to ensure identical visual token counts.

---

## 7. Metrics, Error Handling, & Artifact Schema

### 7.1 Quantitative Evaluation Metrics

All metrics will be calculated independently for Configurations A, B, and C:

1. **Primary Classification Metrics**:
   - Binary Accuracy: $(TP + TN) / (TP + TN + FP + FN)$
   - Macro-F1: $(F1_{\text{humour}} + F1_{\text{non-humour}}) / 2$
   - Class-Specific Metrics: Precision, Recall, and F1 for Humour ($1$) and Non-Humour ($0$).
2. **Ranking & Probabilistic Calibration**:
   - Area Under ROC Curve (ROC-AUC) & Precision-Recall Curve (PR-AUC).
   - Brier Score: $\frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$ (mean squared probability error).
   - Expected Calibration Error (ECE): 10 equal-width probability bins.
3. **Threshold Sensitivity Sweep**:
   - Complete metrics evaluated across thresholds: $\tau \in \{0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70\}$.
4. **Subgroup Disaggregated Accuracy**:
   - Accuracy across each of the 11 strata ($H1\text{--}H5$ and $NH1\text{--}NH6$).
   - Short-circuit frequency and Stage-2 invocation accuracy.

### 7.2 Robust Inference-Failure Handling

Running a large 100-sample multimodal evaluation on local CPU infrastructure carries stability risks. The harness will enforce strict failure-containment rules:

1. **JSON Parsing & Structural Recovery**:
   - All model outputs parsed via `src/vlm/output_parser.py`.
   - If markdown fences or trailing commentary are returned, the parser extracts JSON substrings.
   - If unrepairable, log `json_error: True` and fall back to Stage-1 probability or neutral 0.50 without raising an unhandled exception.
2. **Per-Sample Inference Timeout**:
   - Hard timeout enforced at **1,500 seconds (25 minutes)** per sample stage (incorporating safety margin for vision encoder prefill and thermal throttling).
   - If timeout triggers, kill the generation process, record `prediction: None`, log `timed_out: True`, and record fallback values.
3. **Memory Management (OOM Prevention)**:
   - Explicit garbage collection (`gc.collect()`) after each sample.
   - PyTorch cache clearing (`torch.cuda.empty_cache()` if GPU, or buffer cleanup if CPU).
4. **Atomic Checkpointing & State Persistence**:
   - Execution writes results line-by-line to `inference_results.jsonl` immediately upon completing each sample.
   - If execution is interrupted (e.g. power outage, user pause, server restart), the runner inspects `image_filename` entries in the output file and resumes from sample $K+1$ without re-evaluating completed samples.

### 7.3 Output Directory & Artifact Preservation Rules

To guarantee historical artifact integrity:
- **Dedicated Output Directory**: All new outputs will be saved to `results/evaluation_heldout_100/`:
  - `sample_manifest.json`: Complete listing of the 100 selected memes with metadata, strata, and ambiguity flags.
  - `comparison_results.csv`: Row-by-row tabular comparisons across all configurations.
  - `inference_results.jsonl`: Untruncated model responses and raw JSON records.
  - `evaluation_summary.json`: Aggregated metrics, calibration scores, and confusion matrices.
  - `threshold_sweep_heldout.csv`: Tabular metrics across all evaluated decision thresholds.
- **Historical Invariance**: No files in `results/evaluation_twostage_v4/`, `results/evaluation_twostage/`, or existing `reports/` may be overwritten or modified.

---

## 8. Empirical CPU Runtime & Resource Estimation

### 8.1 Observed Historical Timings from v4

In `results/evaluation_twostage_v4/comparison_results.csv`, exact per-sample CPU durations were recorded on the current machine (`Python 3.13` on Windows):

| Mode / Processing Stage | Mean Duration per Sample | Min Observed | Max Observed | Standard Deviation |
|:---|:---:|:---:|:---:|:---:|
| **General Mode (Stage 1 Qwen)** | **563.93 s (~9.40 min)** | 411.44 s | 807.44 s | ~118.2 s |
| **Cultural Mode: Short-Circuited** | **0.42 s (< 0.01 min)** | 0.23 s | 0.62 s | ~0.15 s |
| **Cultural Mode: Stage 2 Invoked** | **851.87 s (~14.20 min)** | 674.85 s | 1,118.59 s | ~195.4 s |
| **Cultural Mode: Overall Average (v4)** | **289.06 s (~4.82 min)** | 0.23 s | 1,118.59 s | ~387.6 s |
| **Combined v4 Pipeline (General + Cultural)** | **852.99 s (~14.22 min)** | 411.67 s | 1,926.03 s | ~452.1 s |

### 8.2 Projected 100-Meme Runtime Scenarios

The total evaluation duration is heavily governed by the **Stage-2 Invocation Rate** (the fraction of memes triggering external cultural retrieval and the second VLM call).

```
Total Time = (N * T_general) + (N_short_circuit * T_fast) + (N_stage2 * T_stage2)
```

| Execution Scenario | Stage-2 Invocation Rate | General Mode CPU Time | Cultural Mode CPU Time | Total CPU Runtime (Hours) | Expected Wall-Clock Duration |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Scenario 1: Low Gating (v4 Rate)** | 27.3% (27/100 memes) | 15.66 hours | 6.39 hours | **22.05 hours** | ~22 to 24 hours |
| **Scenario 2: Moderate Gating (Target)** | **40.0% (40/100 memes)** | **15.66 hours** | **9.46 hours** | **25.12 hours** | **~24 to 26 hours** |
| **Scenario 3: High Gating (Cultural Focus)** | 55.0% (55/100 memes) | 15.66 hours | 13.01 hours | **28.67 hours** | ~28 to 31 hours |
| **Scenario 4: Two Independent Configurations (A + B)** | Moderate (40%) | 15.66 hours | 9.46 hours | **25.12 hours** | ~25 hours |
| **Scenario 5: Three Configurations (A + B + C)** | Moderate (40%) | 15.66 hours | 18.92 hours | **34.58 hours** | ~34 to 38 hours |

### 8.3 Sources of Runtime Uncertainty
1. **Visual Token Complexity**:
   Memes with dense fine-print text (e.g. multi-line news articles) generate more visual tokens, increasing prefill attention time by up to 35%.
2. **Thermal Throttling**:
   Over a continuous 24-hour CPU run, sustained CPU package temperatures may induce clock throttling, reducing effective throughput by 10% to 20%.
3. **OS Background Interruptions**:
   Windows scheduled maintenance, defender scans, or power events can delay processing.
4. **Recommended Execution Protocol**:
   Due to the 24+ hour runtime, execution **must not be attempted in a single fragile command**. It must be run through an automated chunked script evaluating **blocks of 10 memes**, verifying JSONL flushes between chunks.

---

## 9. Actionable Step-by-Step Implementation Roadmap

When authorized by the user to proceed with execution, the evaluation should follow this ordered roadmap:

```
[Step 1: Manifest Generation]
  ├── Script: scripts/create_heldout_manifest.py
  ├── Inputs: data/processed/test.csv, QUARANTINED_HISTORICAL_MEMES
  ├── Filter: Balanced 50/50, 11 strata, zero overlap
  └── Output: results/evaluation_heldout_100/sample_manifest.json

[Step 2: Pre-Inference Audit & Human Ambiguity Review]
  ├── Script: scripts/audit_heldout_manifest.py
  ├── Verify: Image existence, image hashes, OCR readability
  └── Annotate: Flag ambiguity_status without altering ground truth

[Step 3: Chunked Batch Inference (Config A & B)]
  ├── Script: scripts/run_heldout_evaluation.py --chunk-size 10
  ├── Resume capability: Checks inference_results.jsonl on restart
  └── Artifact: results/evaluation_heldout_100/inference_results.jsonl

[Step 4: Comprehensive Metric & Sweep Analysis]
  ├── Script: scripts/evaluate_heldout_results.py
  ├── Output: comparison_results.csv, evaluation_summary.json
  └── Compute: Accuracy, F1, ROC-AUC, ECE, Subgroup breakdowns

[Step 5: Final Report Generation]
  └── Output: reports/heldout_evaluation_v5_results.md
```

---

## 10. Audit Integrity & Artifact Verification Statement

### 10.1 Exact Files Read
1. `results/evaluation_twostage_v4/comparison_results.csv`
2. `results/evaluation_twostage_v4/inference_results.jsonl`
3. `reports/v4_error_analysis.md`
4. `reports/v4_zero_inference_analysis.md`
5. `reports/v4_threshold_sweep_verified.md`
6. `reports/drishtikon_section144_mitigation_validation.md`
7. `data/processed/train.csv`
8. `data/processed/validation.csv`
9. `data/processed/test.csv`
10. `data/processed/blind_test.csv`

### 10.2 Statement of Zero Modification
- **Code & Configurations**: No code in `src/`, `configs/`, or `scripts/` was modified.
- **Data & Labels**: No datasets in `data/`, labels, or annotations were modified.
- **Historical Reports**: All pre-existing markdown reports and evaluation directories remain untouched.
- **Inference**: Zero VLM calls and zero application services were started.
