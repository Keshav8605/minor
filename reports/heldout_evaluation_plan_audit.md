# Methodological and Execution-Safety Audit: Held-Out Evaluation Plan (`N=100`)

**Target Document**: `reports/heldout_evaluation_plan.md`  
**Audit Type**: Read-Only Methodological, Statistical, & Execution-Safety Audit  
**Project**: Drishtikon Multi-Modal Hindi/Hinglish Meme Humour Detection  
**Auditor**: Antigravity Assistant  
**Date**: October 20, 2026  
**Status**: Formal Audit Complete — Methodological Corrections & Safety Hardening Required Prior to Manifest Finalization  

---

## Executive Summary

A comprehensive, read-only audit of `reports/heldout_evaluation_plan.md` was conducted against the underlying Memotion 3 datasets (`train.csv`, `validation.csv`, `test.csv`, `blind_test.csv`), historical evaluation artifacts (`results/evaluation_twostage_v4/`), and the codebase execution infrastructure (`src/evaluation/checkpoint_manager.py`, `scripts/run_controlled_evaluation.py`, `run_robust_eval.py`, `src/vlm/inference.py`).

The evaluation plan is conceptually sound, addressing key statistical deficiencies of prior small runs ($N=11$) by proposing a balanced 50:50 held-out evaluation with multi-dimensional stratification. However, this audit identified **three high-severity methodological/safety hazards** and **four operational constraints** that must be resolved before finalizing the 100-meme manifest:

1. **High-Severity Bug in Legacy Timeout Handler (`run_robust_eval.py`)**: The fallback script hardcodes `"prediction": 0` on hung/timed-out inferences, silently converting execution failures into class 0 (`Non-Humorous`) predictions and distorting accuracy and recall metrics.
2. **Substratum Quota Infeasibility (`NH3`)**: The plan allocates 10 memes to `NH3 (Factual, Educational, & Historical Graphics)`. Empirical audit of the 91 clean non-humorous candidates reveals only 3–4 genuine educational graphics; rigid enforcement will either stall sampling or force misclassification.
3. **Perceptual Leakage Confirmed & Missing Dependency**: Auditing the 680 candidate memes against the 20 quarantined historical memes uncovered **15 perceptual hash collisions ($d \le 3$)**, including **2 exact visual duplicates ($d=0$)**. While the plan rightly proposed perceptual hash deduplication, the required package `imagehash` is **not installed** in the Python environment.
4. **Tight Timeout Buffer (1,200s)**: Historical run `train_4136.jpg` required **1,118.59s** in cultural mode. A 1,200s hard timeout leaves an unsafe buffer of only 81.4s (6.8%) against thermal throttling.

---

## 1. Audit Verification Matrix

| # | Verification Question | Audit Verdict | Severity | Key Finding / Required Correction |
|:---:|:---|:---:|:---:|:---|
| **1** | **Subgroup Quota Feasibility** | **Partially Feasible** | **High** | Humorous strata ($N=50$) are 100% feasible from 589 candidates. Non-humorous pool has 91 candidates (plan erroneously stated 81 in §2.2). However, `NH3` (quota 10) only contains 3–4 true factual graphics. Quotas must be rebalanced. |
| **2** | **Deduplication Enforcement** | **Confirmed Critical & Dependency Missing** | **High** | Found 15 perceptual hash collisions ($d \le 3$), including 2 identical images ($d=0$) of quarantined memes (`train_6993` vs `train_2837`, `train_5575` vs `train_4199`). However, `imagehash` is not installed; manifest generator must use pure PIL/numpy. |
| **3** | **Multi-Config Isolation (A, B, C)** | **Feasible via Directory Separation** | **Medium** | Existing `CheckpointManager` and runner hardcode two conditions (`general`, `cultural`). Config C cannot be merged into the same checkpoint file. Each config must evaluate into an isolated output directory. |
| **4** | **Runtime Estimation (25.13h)** | **Mathematically Verified** | **Low** | Empirical data from v4 yields 24.50h for 40% Stage-2 gating ($15.66\text{h}$ General + $8.83\text{h}$ Cultural). Three-tier range established: Best-case **16.09h**, Expected **24.50h**, Conservative **30.71h** (A+B) / **44.5h** (A+B+C). |
| **5** | **1,200s Timeout Compatibility** | **Incompatible / Unsafe** | **High** | `train_4136.jpg` ran for 1,118.59s. A 1,200s timeout has only an 81.4s safety margin. Furthermore, vision-encoder prefill is unmonitored by HuggingFace `StoppingCriteria`. Timeout must be relaxed to 1,500s. |
| **6** | **Interruption & Prediction Integrity** | **Vulnerable to Class 0 Corruption** | **Critical** | CheckpointManager correctly tracks completed pairs via JSONL. However, `run_robust_eval.py` line 66 injects `"prediction": 0` on hung tasks! Failures must be recorded as `prediction: None` to avoid corrupting metric calculations. |
| **7** | **Consensus Subset & Blind Ambiguity** | **Methodologically Sound** | **Low** | Ambiguity flagging must occur strictly pre-inference and blind to model predictions. Ground-truth labels remain untouched; dual metrics computed via metadata masks. Zero threshold tuning on test data. |

---

## 2. In-Depth Verification Findings

### 2.1 Subgroup Quota Feasibility & Data Distribution

#### Dataset Split Inventory
The candidate pool was verified across all files in `data/processed/`:
- `data/processed/train.csv`: 5,600 rows (4,792 humorous, 808 non-humorous). Zero overlap with past evaluations.
- `data/processed/validation.csv`: 700 rows (599 humorous, 101 non-humorous). Zero overlap with past evaluations. Reserved strictly for development and hyperparameter tuning.
- `data/processed/blind_test.csv`: 1,500 rows (`image_filename`, `image_url`, `ocr`). Completely unlabelled test set; unusable for supervised evaluation.
- `data/processed/test.csv`: 700 rows (599 humorous, 101 non-humorous). Contains all 20 historical evaluation memes.

#### Candidate Pool Arithmetic Correction
In `reports/heldout_evaluation_plan.md` §2.2, the plan stated:
> *"After permanently excluding the 20 historical evaluation memes, exactly 680 pristine, unexamined memes remain in test.csv (81 non-humorous, 589 humorous)."*

- **Audit Verification**:
  The 20 quarantined historical evaluation memes in `test.csv` consist of **10 humorous** and **10 non-humorous** memes:
  $$\text{Quarantined: } N_{\text{humour}} = 10, \quad N_{\text{non-humour}} = 10$$
  Subtracting the quarantined memes from `test.csv` (700 rows):
  $$N_{\text{humour, clean}} = 599 - 10 = 589$$
  $$N_{\text{non-humour, clean}} = 101 - 10 = \mathbf{91} \quad (\text{not } 81)$$
  $$\text{Total clean candidates} = 589 + 91 = 680$$
  *Finding*: There are 91 clean non-humorous candidates available, providing 10 more candidates than the plan assumed.

#### Humorous Strata Feasibility ($N=50$)
Auditing the 589 clean humorous candidates in `test.csv` demonstrates large surpluses for all 5 humorous strata:
- **H1 (High Cultural Dependency)**: 299 candidates match `CATEGORY_KEYWORDS` (Quota: 15). $\to$ **Feasible (19.9x surplus)**.
- **H2 (Hinglish Daily Life & Banter)**: 130 candidates contain Hinglish markers (`bhai`, `yaar`, `padhai`, `jugaad`) (Quota: 15). $\to$ **Feasible (8.7x surplus)**.
- **H3 (Political Satire & Commentary)**: 484 candidates marked `sarcastic in ['general', 'twisted_meaning', 'very_twisted']` (Quota: 10). $\to$ **Feasible (48.4x surplus)**.
- **H4 (Dark Humour & Edgy Comedy)**: 225 candidates marked `offensive in ['slight', 'very_offensive', 'hateful_offensive']` (Quota: 5). $\to$ **Feasible (45.0x surplus)**.
- **H5 (Universal / General Comedy)**: 230 candidates with no cultural keywords and no Hinglish markers (Quota: 5). $\to$ **Feasible (46.0x surplus)**.

#### Non-Humorous Strata Feasibility & The NH3 Friction ($N=50$)
Selecting 50 non-humorous memes from 91 candidates represents an overall sampling rate of 54.9%. However, when mapping the 91 candidates into the plan's 6 specific negative strata:
- **NH2 (Motivational Quotes)**: 24 candidates with `motivational == 'motivational'` (Quota: 12). $\to$ **Feasible**.
- **NH4 (Sarcastic Non-Humour)**: 29 candidates with `sarcastic in ['twisted_meaning', 'very_twisted']` (Quota: 8). $\to$ **Feasible**.
- **NH5 (Hostile / Aggressive)**: 22 candidates with `offensive in ['very_offensive', 'hateful_offensive']` (Quota: 5). $\to$ **Feasible**.
- **NH1 (Serious News & Tragedy)**: 17 candidates match public order/tragedy/news keywords (Quota: 12). $\to$ **Feasible**.
- **NH6 (Commercial Spam / Ads)**: 8 candidates contain promotional/contact text (Quota: 3). $\to$ **Feasible**.
- **NH3 (Factual, Educational, & Historical Graphics)**: **Only 3 to 4 candidates** match educational/infographic content (Quota: 10). $\to$ **INFEASIBLE as currently specified**.

*Root Cause*: Memotion 3 is sourced from Twitter and Instagram meme feeds. Ordinary non-humorous social posts are predominantly personal complaints, celebrity fan tributes, Bollywood lyrics cards, and sports banter labelled `not_funny`. True science/history infographics are rare.

*Required Rebalancing*:
Reallocate the 10-sample quota of NH3 across the natural distribution of non-humorous content:
$$\begin{array}{l|c|c|l}
\hline
\textbf{Stratum} & \textbf{Plan Quota} & \textbf{Recommended Quota} & \textbf{Selection Criteria} \\
\hline
\text{NH1: Serious News / Current Affairs} & 12 & \mathbf{12} & \text{News, COVID, legal orders, government announcements} \\
\text{NH2: Motivational / Quotes} & 12 & \mathbf{12} & \text{motivational == 'motivational', inspirational text} \\
\text{NH3: Factual / Informational Graphics} & 10 & \mathbf{4} & \text{Statistics, infographics, educational cards} \\
\text{NH4: Sarcastic Non-Humour} & 8 & \mathbf{10} & \text{sarcastic in twisted/very\_twisted, cynical non-humour} \\
\text{NH5: Hostile / Aggressive Statements} & 5 & \mathbf{6} & \text{offensive in very/hateful, provocative statements} \\
\text{NH6: Commercial & Social Media Spam} & 3 & \mathbf{6} & \text{Advertisements, channel handles, spam flyers} \\
\hline
\textbf{Total Non-Humorous} & \mathbf{50} & \mathbf{50} & \text{Sum = 50} \\
\hline
\end{array}$$

---

### 2.2 Deduplication Audit & Perceptual Collisions

An empirical deduplication script was executed across all 680 candidate memes in `test.csv` against the 20 quarantined historical memes (`QUARANTINED_HISTORICAL_MEMES`):
- **Exact File Hash (MD5 / SHA-256)**: 0 collisions.
- **Normalized OCR Levenshtein Similarity ($>85\%$)**: 0 collisions.
- **Perceptual Image Hash (Average Hash, Hamming Distance $d \le 3$)**: **15 collisions discovered**!

#### Concrete Perceptual Collisions Found
| Candidate Meme in `test.csv` | Quarantined Historical Meme | Hamming Distance ($d$) | Image Relationship |
|:---|:---|:---:|:---|
| `train_6993.jpg` | `train_2837.jpg` | **0** | **Exact Visual Duplicate** (Re-encoded Twitter screenshot) |
| `train_5575.jpg` | `train_4199.jpg` | **0** | **Exact Visual Duplicate** (Same meme template, identical layout) |
| `train_181.jpg` | `train_2837.jpg` | 1 | Near-duplicate image card |
| `train_1165.jpg` | `train_2837.jpg` | 1 | Near-duplicate image card |
| `train_3747.jpg` | `train_5723.jpg` | 1 | Same template variant |
| `train_627.jpg` | `train_5723.jpg` | 2 | Same template variant |
| `train_3686.jpg` | `train_6346.jpg` | 2 | Same template variant |
| `train_4249.jpg` | `train_2000.jpg` | 3 | High visual structural overlap |

*Implications*:
1. If deduplication relied only on filename matching, `train_6993.jpg` and `train_5575.jpg` could have entered the held-out test set, creating a direct data leak of memes evaluated in v3/v4!
2. **Environment Constraint**: The Python environment does **not** have the `imagehash` package installed (`ModuleNotFoundError: No module named 'imagehash'`). However, `PIL` (11.3.0) and `numpy` (2.2.3) are installed. The manifest creation script (`scripts/create_heldout_manifest.py`) must implement an in-line 8x8 average hash using `PIL.Image` and `numpy` rather than importing an uninstalled library.

---

### 2.3 Multi-Configuration Protocol & Checkpoint Isolation

The plan defines three configurations:
- **Config A**: Frozen General Baseline (Stage 1 only, no cultural retrieval).
- **Config B**: Frozen Cultural Pipeline (Two-stage gating with Section 144 mitigated DRISHTIKON).
- **Config C**: Targeted Prompt Refinement (Anti-absurdity news filter + Hinglish phonetic prompt).

#### Code Architecture Limitations
In `src/evaluation/checkpoint_manager.py` and `scripts/run_controlled_evaluation.py`:
1. The inference plan hardcodes pairs: `(idx, "general")` and `(idx, "cultural")` (lines 301–303).
2. The summary generator `_generate_final_outputs` merges exactly two conditions: `gen_r` and `cult_r` into `comparison_results.csv` (lines 569–617).
3. Checkpoint tracking sums `general_completed` and `cultural_completed` (lines 205–207).

#### Safety Recommendation: Directory-Isolated Evaluations
Attempting to force Config C into the existing `(general, cultural)` pairing within the same checkpoint directory would break backward compatibility and risk checkpoint corruption.
Instead, each configuration must run into its own dedicated output directory, referencing the single immutable manifest:
```
results/
├── evaluation_heldout_100/
│   └── sample_manifest.json          <-- Shared immutable 100-sample manifest
├── evaluation_heldout_100_config_a/   <-- Frozen General Baseline outputs
│   ├── evaluation_checkpoint.json
│   ├── inference_results.jsonl
│   └── comparison_results.csv
├── evaluation_heldout_100_config_b/   <-- Frozen Cultural Baseline (Mitigated)
│   ├── evaluation_checkpoint.json
│   ├── inference_results.jsonl
│   └── comparison_results.csv
└── evaluation_heldout_100_config_c/   <-- Targeted Prompt Refinement Candidate
    ├── evaluation_checkpoint.json
    ├── inference_results.jsonl
    └── comparison_results.csv
```
A lightweight comparison script (`scripts/compare_heldout_configs.py`) can then load all three `comparison_results.csv` files, compute delta metrics, and generate the final three-way comparison table.

---

### 2.4 CPU Runtime & Workload Verification

#### Empirical Timing Audit from `results/evaluation_twostage_v4/`
The 11 memes evaluated in v4 yielded the following exact timings on the local CPU (`Python 3.13` on Windows):

| Processing Mode / Component | Count ($N$) | Mean Duration | Min Observed | Max Observed | Standard Deviation |
|:---|:---:|:---:|:---:|:---:|:---:|
| **General Mode (Stage 1 Qwen)** | 11 | **563.93 s (~9.40 min)** | 411.44 s | 807.44 s | 112.58 s |
| **Cultural Mode: Short-Circuited** | 7 (63.6%) | **0.46 s (< 0.01 min)** | 0.23 s | 0.97 s | 0.25 s |
| **Cultural Mode: Stage 2 Invoked** | 4 (36.4%) | **794.12 s (~13.24 min)** | 620.84 s | 1,118.59 s | 227.14 s |
| **Cultural Mode Overall** | 11 | **289.06 s (~4.82 min)** | 0.23 s | 1,118.59 s | 418.80 s |
| **Combined Paired Duration** | 11 | **852.99 s (~14.22 min)** | 411.67 s | 1,926.03 s | 452.10 s |

*(Note: In the plan §8.1, Stage 2 duration was listed as 851.87s based on an initial 3-sample subset. When auditing all 4 Stage-2 samples—`train_6346` [762.2s], `train_5723` [674.9s], `train_4136` [1118.6s], `train_5890` [620.8s]—the true mean is 794.12s).*

#### 100-Meme Runtime Workload Calculation
For evaluating Config A and Config B across 100 memes with an expected 40% Stage-2 gating rate:
$$\text{Time}_{\text{Config A}} = 100 \times 563.93\text{ s} = 56,393\text{ s} = \mathbf{15.66\text{ hours}}$$
$$\text{Time}_{\text{Config B}} = (60 \times 0.46\text{ s}) + (40 \times 794.12\text{ s}) = 27.6\text{ s} + 31,764.8\text{ s} = 31,792.4\text{ s} = \mathbf{8.83\text{ hours}}$$
$$\text{Total CPU Time (A + B)} = 56,393\text{ s} + 31,792\text{ s} = 88,185.4\text{ s} = \mathbf{24.50\text{ hours}}$$

The plan’s stated **25.13-hour estimate** is confirmed within 2.5% of the recalculated empirical mean.

#### Three-Tier Projection Matrix
| Scenario | Stage-2 Gating Rate | General Time | Cultural Time | Config A+B Time | Config A+B+C Time | Key Assumptions |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Best-Case** | 27.3% (27/100) | 11.43 h | 4.66 h | **16.09 hours** | **23.5 hours** | Low visual token count (min observed timings), no thermal throttling. |
| **Expected** | 40.0% (40/100) | 15.66 h | 8.83 h | **24.50 hours** | **34.5 hours** | Mean empirical timings, 40% Stage-2 gating rate. |
| **Conservative** | 50.0% (50/100) | 18.01 h | 12.70 h | **30.71 hours** | **44.5 hours** | High visual tokens, +15% thermal clock throttling over 24+ hour CPU run. |

---

### 2.5 Timeout Compatibility & The Prefill Decoupling Vulnerability

#### The 1,200-Second Timeout Buffer Risk
In `results/evaluation_twostage_v4/inference_results.jsonl`, sample `train_4136.jpg` recorded:
- Cultural inference duration: **1,118.59 seconds (18.64 minutes)**!
- Generated token count: 349 tokens.
- Output: Clean JSON, `timed_out: False`.

$$\text{Margin under 1,200s timeout} = 1,200.0\text{ s} - 1,118.59\text{ s} = \mathbf{81.41\text{ seconds (6.8\% margin)}}$$

If a complex meme in the 100-sample set generates slightly more visual patches or suffers a 10% thermal slowdown, its duration would reach $1,118.59 \times 1.10 = 1,230.45\text{s}$, triggering a false timeout on a healthy inference!

#### Codebase Implementation Discrepancy
1. **HuggingFace StoppingCriteria Limitation**:
   In `src/vlm/inference.py` lines 31–65 (`DiagnosticStoppingCriteria`):
   `StoppingCriteria.__call__` is invoked **only during autoregressive token generation**!
   It does **not** run during the vision-encoder prefill phase. If vision prefill takes 500s, `DiagnosticStoppingCriteria` has not started counting down `generation_timeout_seconds`.
2. **Subprocess Killer Discrepancy**:
   In the legacy execution wrapper `run_robust_eval.py`, line 8 hardcodes:
   `TIMEOUT_S = 900  # 15 minutes max per inference`
   If `run_robust_eval.py` were executed, `train_4136.jpg` (1,118.59s) would have been killed and skipped at 900 seconds!

*Required Correction*:
- Set the per-sample outer process timeout to **1,500 seconds (25 minutes)**.
- Set internal decoding timeout to **900 seconds**.
- Align all wrapper scripts (`run_eval_loop.ps1`, `run_robust_eval.py`, and `scripts/run_controlled_evaluation.py`) to the identical 1,500s threshold.

---

### 2.6 Critical Execution-Safety Bug: The Class 0 Fallback Hazard

#### Code Inspection of `run_robust_eval.py`
In `run_robust_eval.py` lines 58–71 (`skip_current_hung`), when a process times out:
```python
dummy_record = {
    "sample_index": s_idx,
    "condition": cond,
    "meme_id": f"sample_{s_idx:03d}_{img_data['image_filename']}",
    "ocr_text": "Skipped",
    "ground_truth": img_data['ground_truth'],
    "prediction": 0,  # <--- CRITICAL BUG: HARDCODED TO CLASS 0!
    "humor_probability": None,
    "non_humor_probability": None,
    "reasoning": "SKIPPED: Model hang/deadlock detected by wrapper timeout.",
    "json_error": "HUNG_PROCESS",
    "timed_out": True,
}
with open(jsonl_path, "a", encoding="utf-8") as f:
  f.write(json.dumps(dummy_record) + "\n")
```

#### Downstream Metric Corruption
In `scripts/run_controlled_evaluation.py` lines 629–633:
```python
gen_valid_mask = res_df["general_prediction"].notna() & res_df[
    "general_prediction"
].isin([0, 1])
y_true_gen = res_df.loc[gen_valid_mask, "ground_truth"].tolist()
y_gen = res_df.loc[gen_valid_mask, "general_prediction"].astype(int).tolist()
```
Because `"prediction": 0` is in `[0, 1]`, the metric evaluation logic treats this hung sample as a valid prediction of `Non-Humorous`!
- If the meme’s ground truth was `0`, the system registers a **True Negative**, artificially boosting accuracy!
- If the meme’s ground truth was `1`, the system registers a **False Negative**, falsely penalizing humor recall!

#### Checkpoint & Resumption Correctness
- `CheckpointManager` itself functions properly: it loads `inference_results.jsonl` and tracks `(sample_index, condition)` pairs. Re-running with `--continue` skips already persisted samples without re-evaluating them.
- *Required Safety Patch*: Any execution script for the 100-sample evaluation must explicitly persist `"prediction": None` and `"timed_out": True`. Downstream evaluation must strictly filter out `None` predictions from standard accuracy/F1 and report a separate `Failure/Timeout Rate`.

---

### 2.7 Ambiguity Management & Scientific Integrity

The ambiguity management protocol specified in §5.3 of the plan was evaluated for methodological soundness:
1. **Pre-Inference Quarantine**: The human review of ambiguous memes must occur **prior to model execution**. Annotators must evaluate memes blind to model predictions to prevent post-hoc bias (i.e. labeling model mistakes as "ambiguous").
2. **Zero Ground-Truth Modification**: The official `is_humorous` column in `sample_manifest.json` will strictly retain its original binary value (0 or 1).
3. **Disaggregated Dual Reporting**:
   - **Full Benchmark ($N=100$)**: Primary, unedited authoritative metrics.
   - **Clean Consensus Subset ($N \approx 75\text{--}80$)**: Evaluates model performance on unambiguous memes, isolating core visual-linguistic reasoning from crowdsourcing noise.
4. **Zero Test-Set Threshold Tuning**: Threshold sweeps on the 100-meme test set must be reported strictly as an analytical calibration verification (evaluating latent logit alignment). Any operational decision threshold must be selected on `data/processed/validation.csv` and applied without modification.

---

## 3. Required Modifications to `heldout_evaluation_plan.md`

Before authorizing manifest creation and evaluation execution, the following revisions must be adopted:

```markdown
1. §2.2 Data Inventory:
   - Correct clean non-humorous count from 81 to 91 (101 total - 10 quarantined = 91).

2. §3.2 Deduplication:
   - Replace reliance on external `imagehash` library with a self-contained PIL/numpy 8x8 average hash implementation.
   - Quarantining must explicitly check against the 15 perceptual collision memes identified in this audit.

3. §4.2 Negative Class Stratification:
   - Rebalance non-humorous quotas:
     • NH1 (Serious News): 12
     • NH2 (Motivational): 12
     • NH3 (Factual Graphics): 4 (reduced from 10)
     • NH4 (Sarcastic Non-Humour): 10 (increased from 8)
     • NH5 (Hostile Statements): 6 (increased from 5)
     • NH6 (Commercial / Spam): 6 (increased from 3)

4. §6.1 Directory & Config Isolation:
   - Mandate dedicated output directories for each configuration (Config A, Config B, Config C) rather than sharing a single results directory.

5. §7.2 Timeout & Error Handling:
   - Increase per-sample hard process timeout from 1,200s to 1,500s (25 minutes).
   - Prohibit setting `prediction: 0` on hung/failed inferences; enforce `prediction: None` and `timed_out: True`.
   - Require dual metric evaluation (Valid Completion Rate vs Metric on Valid Set).
```

---

## 4. Audit Integrity Statement

### 4.1 Exact Files Read
1. `reports/heldout_evaluation_plan.md`
2. `data/processed/train.csv`
3. `data/processed/validation.csv`
4. `data/processed/test.csv`
5. `data/processed/blind_test.csv`
6. `results/evaluation_twostage_v4/comparison_results.csv`
7. `results/evaluation_twostage_v4/inference_results.jsonl`
8. `src/evaluation/checkpoint_manager.py`
9. `src/evaluation/prediction_store.py`
10. `scripts/run_controlled_evaluation.py`
11. `run_robust_eval.py`
12. `src/vlm/inference.py`

### 4.2 Statement of Zero Modification
- **Datasets & Labels**: No datasets in `data/`, labels, or CSV files were modified.
- **Source Code**: No code in `src/`, `configs/`, or `scripts/` was modified.
- **Historical Reports & Results**: All existing files in `reports/` and `results/` remain completely untouched.
- **Manifest Status**: No 100-sample manifest was created or finalized.
- **Inference**: Zero VLM calls were executed and no application services were started.
