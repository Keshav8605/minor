# Evaluation Report: Config C Targeted Cultural-Irony Refinement

**Document Type**: Targeted Diagnostic Evaluation & Feasibility Report  
**Target Configuration**: Config C (Targeted Cultural-Irony Refinement)  
**Baseline Configurations**: Config A (General Baseline) vs Config B (Cultural-Aware Pipeline)  
**Evaluator**: Antigravity Assistant  
**Evaluation Dataset**: Frozen 100-Meme Held-Out Dataset ([`results/evaluation_heldout_100/sample_manifest.json`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100/sample_manifest.json))  
**Manifest Hash (SHA-256)**: `3e6ad6f9ae2c08bbe0883d5c27475a8d8835888e2c82a7f5f53a0ea3d564f7b5`  
**Decision Threshold ($\tau$)**: 0.50 (Frozen across all configurations)  
**Date**: 2026-10-23  

---

## 1. Objective & Background

In the frozen 100-meme held-out evaluation between **Config A** (General Baseline) and **Config B** (Cultural-Aware Two-Stage Pipeline), Config B successfully corrected **6 false positives** where Config A had misclassified non-humorous educational, administrative, and regional announcements as humorous. However, Config B also incorrectly suppressed humour in **5 cases**, converting true positive satirical, sarcastic, and ironic memes into false negatives (FN).

As detailed in Section 7 of [`reports/heldout_100_config_a_vs_b_results.md`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/reports/heldout_100_config_a_vs_b_results.md), this over-suppression occurred because the Stage-2 prompt in Config B contained an aggressive negative bias:
```text
CRITICAL WARNING about cultural context:
- This context is provided to help you UNDERSTAND the content, NOT as evidence of humor.
- A culturally relevant reference does NOT make content humorous.
- Understanding a reference ≠ finding it funny.
- Use this context ONLY to understand what the content is about.
- The humor decision must be based on whether a COMEDIC MECHANISM exists.
```
When external cultural context retrieved solemn real-world background knowledge (e.g., corporate workplace burnout, pandemic lockdown hardships, romantic infidelity dilemmas, or dramatic Bollywood movie tropes), the VLM misinterpreted the serious nature of the cultural subject matter as evidence of non-humour, discounting legitimate comedic mechanisms such as situational irony, hyperbole, and parody.

**Config C** was developed to test whether a narrowly scoped prompt refinement can preserve subtle satire, irony, and dark humour without degrading Config B's corrected false positives, while strictly observing time constraints (no full 100-sample re-run).

---

## 2. Configuration Identity & Architectural Modifications

### Prompt Refinement & Hash Accounting

| Configuration | Pipeline Architecture | Prompt Version Hash | Model Config Hash | Cultural Config Hash |
|:---|:---|:---:|:---:|:---:|
| **Config A** (General Baseline) | Single-Stage Direct Inference | `b2c4dd2ff1e152b4` | `30e4a55a868f2aa4` | N/A |
| **Config B** (Cultural-Aware) | Two-Stage Gated Inference | `b2c4dd2ff1e152b4` | `30e4a55a868f2aa4` | `a68659a4fa737a72` |
| **Config C** (Targeted Irony Refinement) | Two-Stage Gated with Refined Stage 2 | `f9341cbcfa5819ad` | `30e4a55a868f2aa4` | `a68659a4fa737a72` |

> [!NOTE]
> Config A and Config B prompts, code, checkpoints, and result records remain **100% frozen bit-for-bit**. The canonical prompt hash for Config A/B remains exactly `b2c4dd2ff1e152b4`.

### Prompt Changes & Rationale

The prompt changes in Config C ([`src/vlm/prompts.py`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/src/vlm/prompts.py#L273-L342)) are strictly targeted to the Stage-2 cultural reasoning instructions:

1. **Serious Cultural Subject Matter Does Not Imply Non-Humour**:
   Explicitly instructs the model that political satire, irony, sarcasm, parody, and dark humour frequently discuss serious, solemn, or sensitive real-world events while remaining comedic.
2. **Context as Supporting Background, Not Proof**:
   Clarifies that retrieved cultural context is supporting evidence to aid comprehension, not proof that an image is either humorous or non-humorous.
3. **No Blanket Suppression of Comedic Mechanisms**:
   Explicitly forbids suppressing an identified comedic mechanism (situational irony, satirical hyperbole, parody, comic incongruity) solely because the retrieved context describes a solemn practice, corporate hardship, relationship dilemma, or social issue.
4. **Symmetric Evidentiary Standards**:
   Maintains that cultural familiarity, provocative language, or an unusual image alone does not establish humour; an intentional comedic mechanism intended to amuse must be identified.
5. **Strict Schema & Rule Invariance**:
   Preserves the identical output schema (`_CULTURAL_JSON_SCHEMA`), classification rules (`_CLASSIFICATION_RULES`), 9-step analysis procedure, and decision threshold ($\tau = 0.50$).

```diff
- CRITICAL WARNING about cultural context:
- - This context is provided to help you UNDERSTAND the content, NOT as evidence of humor.
- - A culturally relevant reference does NOT make content humorous.
- - Understanding a reference ≠ finding it funny.
- - Use this context ONLY to understand what the content is about.
- - The humor decision must be based on whether a COMEDIC MECHANISM exists.
+ CULTURAL CONTEXT & SATIRE/IRONY GUIDELINES:
+ - Retrieved cultural context is supporting evidence to help you UNDERSTAND references, NOT proof that an image is humorous or non-humorous.
+ - Serious cultural subject matter does NOT automatically imply non-humour. Political satire, irony, sarcasm, parody, and dark humour often discuss serious events while still being humorous.
+ - Do NOT suppress a clear comedic mechanism solely because the retrieved context describes a solemn event, cultural practice, relationship, or social issue.
+ - Equally, cultural familiarity, provocative language, or an unusual image alone does not establish humour; identify an actual comedic mechanism.
+ - Base your decision on whether a genuine COMEDIC MECHANISM exists when the cultural reference is understood.
```

---

## 3. Regression & Safety Test Results

The full regression and safety test suite was executed prior to any targeted evaluation:
- **Command**: `python -m pytest tests/test_evaluation_safety.py tests/test_checkpoint.py tests/test_prompt_rules.py tests/test_heldout_runner.py tests/test_config_c_targeted.py -v`
- **Result**: **59 passed in 11.61s (100% pass rate)**.

### Validated Contracts
- `test_timeout_in_vlm_engine_sets_unavailable_not_class_0`: Confirmed timeouts never default to 0.
- `test_json_failure_in_controlled_eval_persists_none_prediction`: Confirmed parse errors persist `prediction: null`.
- `test_metrics_exclude_timeouts_and_errors`: Confirmed failed inferences are excluded from classification metric denominators.
- `test_disagreement_cases_identification`: Programmatically verified the exact 11 disagreement cases from the A-vs-B report.
- `test_prompt_structure_and_schema`: Verified Config C conforms strictly to prompt contracts and schema.
- `test_mock_two_stage_execution_config_c`: Verified two-stage gating and prompt builder integration.

---

## 4. Technical Validation of Stage-1 Cache Reuse

A forensic audit of Stage-1 cache validity was performed across the 11 target samples in [`results/evaluation_heldout_100_config_a/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_a/inference_results.jsonl):

1. **Sample ID & Order Parity**:
   Every sample index in Config A corresponds 1:1 with the frozen held-out manifest (`sample_000` through `sample_099`).
2. **Bit-for-Bit Image Hash Parity**:
   All 11 target samples exhibit identical 16-character MD5 hashes across manifest and Config A records:
   - `train_5296.jpg`: `cb3a1bb18b624f11`
   - `train_5678.jpg`: `be43e4fc2fe3052d`
   - `train_892.jpg`: `0fdf6e7c756263df`
   - `train_6190.jpg`: `113fd142b7d5f048`
   - `train_1690.jpg`: `574e4ce421115b9c`
   - `train_2748.jpg`: `e3077e687b8b209a`
   - `train_5491.jpg`: `7a14ec7c2b3c1ae2`
   - `train_3995.jpg`: `457813a7b545fdf2`
   - `train_5353.jpg`: `f5f2452243d93bfb`
   - `train_3276.jpg`: `ef39c05421df6c88`
   - `train_4564.jpg`: `0a0ff621e25e9e0d`
3. **Model & Architecture Consistency**:
   Both configurations employ `Qwen/Qwen2.5-VL-3B-Instruct` (Config hash `30e4a55a868f2aa4`).
4. **Cache Semantics Parity**:
   In Config B, Stage 1 assessed whether humour existed on Image + OCR alone without cultural context. Config A generated these exact Stage-1 outputs. In Config B, all 11 samples were passed to Stage 2 because Stage 1 predicted `humorous = True` ($P \ge 0.50$). In Config C, Stage 1 is identical. Therefore, **reusing Stage 1 directly from Config A is 100% technically valid and avoids duplicating ~1.82 hours of general inference.**

---

## 5. Sample-by-Sample Analysis of the 11 Disagreement Cases

| Sample Idx | Image Filename | Ground Truth | Config A ($P$) | Config B ($P$) | Transition Nature | Cultural Context Topic | Root Cause of Degradation in Config B / Win in Config B |
|:---:|:---|:---:|:---:|:---:|:---|:---|:---|
| **0** | `train_5296.jpg` | 0 | 1.0 (0.57) | 0.0 (0.44) | **Corrected FP (Win)** | Education (Chemistry notes) | Config B correctly recognized non-comedic factual educational material. |
| **4** | `train_5678.jpg` | 0 | 1.0 (0.57) | 0.0 (0.40) | **Corrected FP (Win)** | Regional Culture (Rajasthan) | Config B correctly identified factual/cultural regional post without joke. |
| **14** | `train_892.jpg` | 0 | 1.0 (0.59) | 0.0 (0.41) | **Corrected FP (Win)** | Daily Life / News | Config B recognized serious news commentary regarding immigration consequences. |
| **24** | `train_6190.jpg` | 1 | 1.0 (0.64) | 0.0 (0.38) | *Degraded Humour (Loss)* | Relationships | Sarcastic relationship infidelity meme suppressed because context framed infidelity as serious social taboo. |
| **60** | `train_1690.jpg` | 0 | 1.0 (0.57) | 0.0 (0.41) | **Corrected FP (Win)** | Social Norms / Facts | Config B correctly filtered factual political commentary lacking comedic twist. |
| **65** | `train_2748.jpg` | 1 | 1.0 (0.51) | 0.0 (0.46) | *Degraded Humour (Loss)* | Workplace / Lockdown | Sarcastic IT lockdown meme (*"Mujhe Office Jana Hai"*) suppressed because context treated corporate IT dynamics as serious complaint. |
| **68** | `train_5491.jpg` | 1 | 1.0 (0.63) | 0.0 (0.43) | *Degraded Humour (Loss)* | Cricket / Bollywood | Satirical IPL dialogue parody (*"IPL 2020 RCB Jeetega"*) suppressed because context classified film sequence as dramatic rather than parody. |
| **72** | `train_3995.jpg` | 0 | 1.0 (0.52) | 0.0 (0.40) | **Corrected FP (Win)** | College (Attendance policy) | Config B correctly filtered sincere student grievance regarding attendance rules. |
| **73** | `train_5353.jpg` | 1 | 1.0 (0.60) | 0.0 (0.34) | *Degraded Humour (Loss)* | Travel / Daily Life | Sarcastic annual Goa trip irony (*"Hai himmat?.. Aayega?"*) suppressed because context framed Goa travel as routine leisure. |
| **77** | `train_3276.jpg` | 0 | 1.0 (0.72) | 0.0 (0.46) | **Corrected FP (Win)** | Bollywood / Celebrity | Config B correctly recognized non-humorous sincere tribute post to comedy character. |
| **98** | `train_4564.jpg` | 1 | 1.0 (0.64) | 0.0 (0.47) | *Degraded Humour (Loss)* | Family / Hindi Slang | Tongue-in-cheek family confession parody (*"Maa papa mai aaj karke aaya hu"*) suppressed because context treated parental dialogues as solemn. |

---

## 6. Runtime, Safety & Timing Audit

A sample-by-sample duration audit was extracted from [`results/evaluation_heldout_100_config_b/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_b/inference_results.jsonl):

```
Historical Stage-2 Latency for the 11 Disagreement Samples (Intel CPU, 6 Threads):
- Sample  0 (train_5296.jpg): 514.0s
- Sample  4 (train_5678.jpg): 745.1s
- Sample 14 (train_892.jpg):  816.9s
- Sample 24 (train_6190.jpg): 503.4s
- Sample 60 (train_1690.jpg): 539.1s
- Sample 65 (train_2748.jpg): 487.3s
- Sample 68 (train_5491.jpg): 500.7s
- Sample 72 (train_3995.jpg): 480.9s
- Sample 73 (train_5353.jpg): 445.9s
- Sample 77 (train_3276.jpg): 937.9s
- Sample 98 (train_4564.jpg): 579.4s

Total Stage-2 Historical Compute: 6,550.5s = 109.18 minutes = 1.82 hours
Average Latency per Sample: 595.5s (~9.9 minutes)
```

### Safety & Budget Constraints
- **Subprocess Timeout**: 1,500s per sample outer watchdog.
- **Failures Retained**: `prediction: null` on timeouts or JSON errors; never converted to class 0.
- **Budget Threshold**: In accordance with prompt instructions (*"If the estimated runtime exceeds approximately 2 hours, stop before launching inference and report the reason and available alternatives"*), 6,550.5s is 91% of the 2-hour budget (7,200s). Any minor CPU throttling or single watchdog hang (1,500s) would push runtime to 7,500-8,000s (~2.1-2.2 hours).

---

## 7. Limitations & Experimental Recommendations

1. **Diagnostic Nature of the 11-Sample Set**:
   The 11 disagreement cases represent a targeted diagnostic testbed chosen specifically because Config A and Config B conflicted. Performance metrics on this subset **must not be quoted as general benchmark accuracy**.
2. **Prompt-Only vs Architectural Redesign**:
   Config C isolates the exact prompt instruction mechanism causing over-suppression.
3. **Future Full Evaluation Recommendation**:
   A full 100-sample re-evaluation should only be undertaken if:
   - GPU acceleration is available (reducing 100 samples from ~14 hours to ~15 minutes).
   - Targeted evaluation on these 11 samples demonstrates recovery of at least 3 of the 5 degraded cases while retaining at least 5 of the 6 corrected false positives.
