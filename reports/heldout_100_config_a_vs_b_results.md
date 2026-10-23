# Controlled Evaluation Report: Config A (General Baseline) vs Config B (Cultural-Aware Pipeline)

**Document Type**: Forensic Experimental Evaluation Report  
**Date**: 2026-10-22 23:56:49  
**Dataset**: Frozen 100-Meme Held-Out Evaluation Dataset  
**Primary Manifest**: [`results/evaluation_heldout_100/sample_manifest.json`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100/sample_manifest.json)  
**Primary Manifest MD5**: `b63b676298480fc13c3231246a658093`  
**Primary Manifest SHA-256**: `3e6ad6f9ae2c08bbe0883d5c27475a8d8835888e2c82a7f5f53a0ea3d564f7b5`  
**Parity Manifest**: [`data/processed/sample_manifest.json`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/data/processed/sample_manifest.json) (100% Bit-for-Bit Identical)  
**Evaluator**: Antigravity Assistant  
**Hardware Mode**: Intel CPU (6 Execution Threads, CUDA Disabled)  
**Timeout Setting**: 1,500s per sample outer OS subprocess watchdog  

---

## 1. Executive Summary

A controlled, reproducible comparative evaluation of **Config A (General Baseline)** and **Config B (Cultural-Aware Two-Stage Pipeline with Mitigated DRISHTIKON RAG)** was executed across the frozen 100-meme held-out evaluation dataset.

Both configurations consumed the exact same 100 manifest entries in identical sample order, evaluating against unedited Memotion 3 ground-truth labels (50 Humorous, 50 Non-Humorous). In accordance with strict experimental constraints:
- **Zero Training or Fine-Tuning**: Model weights remained frozen; no backpropagation or parameter modification was performed.
- **Zero Threshold Tuning**: The decision threshold was held constant at $\tau = 0.50$ across all evaluations.
- **Isolated Checkpoints**: Config A and Config B results were recorded in distinct directories (`results/evaluation_heldout_100_config_a/` and `results/evaluation_heldout_100_config_b/`).
- **Failure Exclusion**: Timed-out or malformed inferences were retained as unavailable predictions (`prediction: null`) and excluded from classification metric denominators.

---

## 2. Configuration Identity & Architectural Differences

| Attribute | Config A (General Baseline) | Config B (Cultural-Aware) |
|:---|:---|:---|
| **Pipeline Architecture** | Single-Stage Direct Inference | Two-Stage Gated Inference |
| **Model Checkpoint** | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **Model Config Hash** | `30e4a55a868f2aa4` | `30e4a55a868f2aa4` |
| **Prompt Version Hash** | `b2c4dd2ff1e152b4` | `b2c4dd2ff1e152b4` |
| **Cultural Config Hash** | N/A (Disabled) | `a68659a4fa737a72` |
| **External Context Retrieval** | Disabled | Enabled (Project JSON + Mitigated DRISHTIKON RAG) |
| **Stage-1 Gating** | None (All Direct) | Evaluates Stage 1; invokes Stage 2 if ambiguous or cultural |
| **Decision Threshold ($\tau$)** | 0.50 | 0.50 |
| **Output Directory** | `results/evaluation_heldout_100_config_a/` | `results/evaluation_heldout_100_config_b/` |

---

## 3. Overall Performance Summary

| Metric | Config A (General Baseline) | Config B (Cultural-Aware) | Delta (Config B - Config A) |
|:---|:---:|:---:|:---:|
| **Evaluated Valid Samples** | 95 / 100 | 94 / 100 | -1 |
| **Accuracy** | 0.5368 | 0.5532 | 0.0164 |
| **Humour Precision** | 0.5263 | 0.5469 | 0.0206 |
| **Humour Recall** | 0.8333 | 0.7292 | -0.1041 |
| **Humour Macro-F1** | 0.6452 | 0.625 | -0.0202 |
| **Non-Humour Precision** | 0.5789 | 0.5667 | -0.0122 |
| **Non-Humour Recall** | 0.234 | 0.3696 | 0.1356 |
| **Non-Humour Macro-F1** | 0.3333 | 0.4474 | 0.1141 |
| **Mean Inference Duration** | 635.2 s | 373.3 s | -261.9 s |
| **Median Inference Duration** | 533.8 s | 0.4 s | -533.4 s |
| **Timeouts ($> 1,500$ s)** | 2 | 3 | 1 |
| **Parse Errors / Malformed** | 5 | 6 | 1 |

---

## 4. Confusion Matrices

### Config A (General Baseline)
```
                  Predicted Non-Humour (0)   Predicted Humour (1)
Actual Non-Humour (0)         11 (TN)                  36 (FP)
Actual Humour (1)             8 (FN)                  40 (TP)
```

### Config B (Cultural-Aware Pipeline)
```
                  Predicted Non-Humour (0)   Predicted Humour (1)
Actual Non-Humour (0)         17 (TN)                  29 (FP)
Actual Humour (1)             13 (FN)                  35 (TP)
```

---

## 5. Performance Disaggregated by Heuristic Ambiguity Status

> [!NOTE]
> Ambiguity status was determined through automated heuristic screening of known crowdsourcing noise patterns (serious news juxtaposition, cynical sarcasm, taboo edginess), not through an independent human consensus panel.

| Subset | Sample Count | Config A Accuracy | Config B Accuracy | Config A Humour F1 | Config B Humour F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clean Consensus Screened** | 87 | 0.6024 | 0.5976 | 0.7027 | 0.6733 |
| **Borderline Heuristic Screened** | 13 | 0.0833 | 0.25 | 0.1538 | 0.1818 |

---

## 6. Per-Stratum Performance Breakdown

| Stratum ID | Stratum Description | Quota | Config A Valid | Config B Valid | Config A Acc | Config B Acc | Config A F1 | Config B F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H1** | High Cultural Dependency | 15 | 14 | 14 | 0.8571 | 0.7857 | 0.9231 | 0.8800 |
| **H2** | Hinglish Daily Life & Situational Irony | 15 | 15 | 15 | 0.8000 | 0.6667 | 0.8889 | 0.8000 |
| **H3** | Political Satire & Societal Commentary | 10 | 9 | 9 | 0.8889 | 0.6667 | 0.9412 | 0.8000 |
| **H4** | Dark Humour & Edgy Comedy | 5 | 5 | 5 | 0.6000 | 0.6000 | 0.7500 | 0.7500 |
| **H5** | Universal / Non-Cultural Situational Comedy | 5 | 5 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **NH1** | Serious News, Tragedy, & Current Affairs Juxtaposition | 12 | 11 | 12 | 0.1818 | 0.3333 | 0.0000 | 0.0000 |
| **NH2** | Motivational Quotes & Inspirational Sayings | 12 | 11 | 10 | 0.1818 | 0.4000 | 0.0000 | 0.0000 |
| **NH3** | Factual, Educational, & Historical Graphics | 4 | 4 | 4 | 0.7500 | 1.0000 | 0.0000 | 0.0000 |
| **NH4** | Sarcastic Non-Humour (Subtle/Cynical) | 10 | 9 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **NH5** | Hostile / Aggressive Social Media Statements | 6 | 6 | 5 | 0.5000 | 0.6000 | 0.0000 | 0.0000 |
| **NH6** | Commercial Advertisements & Watermarked Spam | 6 | 6 | 6 | 0.1667 | 0.3333 | 0.0000 | 0.0000 |

---

## 7. Sample Disagreements & Prediction Transitions

Total samples with differing predictions between Config A and Config B: **11**

| Sample Index | Image Filename | Ground Truth | Config A Pred ($P$) | Config B Pred ($P$) | Transition Nature | Cultural Category |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| 0 | `train_5296.jpg` | 0 | 1.0 (0.57) | 0.0 (0.44) | **Corrected by Cultural Pipeline (Win)** | education |
| 4 | `train_5678.jpg` | 0 | 1.0 (0.57) | 0.0 (0.40) | **Corrected by Cultural Pipeline (Win)** | regional_culture |
| 14 | `train_892.jpg` | 0 | 1.0 (0.59) | 0.0 (0.41) | **Corrected by Cultural Pipeline (Win)** | No specific cultural category detected |
| 24 | `train_6190.jpg` | 1 | 1.0 (0.64) | 0.0 (0.38) | *Degraded by Cultural Pipeline (Loss)* | relationships |
| 60 | `train_1690.jpg` | 0 | 1.0 (0.57) | 0.0 (0.41) | **Corrected by Cultural Pipeline (Win)** | No specific cultural category detected |
| 65 | `train_2748.jpg` | 1 | 1.0 (0.51) | 0.0 (0.46) | *Degraded by Cultural Pipeline (Loss)* | workplace |
| 68 | `train_5491.jpg` | 1 | 1.0 (0.63) | 0.0 (0.43) | *Degraded by Cultural Pipeline (Loss)* | Bollywood |
| 72 | `train_3995.jpg` | 0 | 1.0 (0.52) | 0.0 (0.40) | **Corrected by Cultural Pipeline (Win)** | college |
| 73 | `train_5353.jpg` | 1 | 1.0 (0.60) | 0.0 (0.34) | *Degraded by Cultural Pipeline (Loss)* | No specific cultural category detected |
| 77 | `train_3276.jpg` | 0 | 1.0 (0.72) | 0.0 (0.46) | **Corrected by Cultural Pipeline (Win)** | Bollywood |
| 98 | `train_4564.jpg` | 1 | 1.0 (0.64) | 0.0 (0.47) | *Degraded by Cultural Pipeline (Loss)* | No specific cultural category detected |

---

## 8. Failure Analysis & Runtime Safety Audit

- **Subprocess Timeouts**: 2 (Config A), 3 (Config B). All hung processes safely killed after 1,500s.
- **Parse Errors**: 5 (Config A), 6 (Config B).
- **Stage 2 Fallbacks to Stage 1**: 0 instances where Stage 2 encountered errors and safely reverted to Stage 1 assessment.

---

## 9. Limitations & Experimental Interpretation

1. **Sample Size Uncertainty ($N = 100$)**: With 100 test memes, standard error for accuracy is approximately $\pm 5\%$. While this held-out set provides a statistically controlled benchmark without data leakage, small differences in performance should not be claimed as absolute proof of universal superiority.
2. **CPU Inference Latency**: The CPU execution speed (~470-520s per sample) necessitates efficient Stage-1 gating in practical deployment.
3. **Recommendation on Config C**: Config B demonstrates positive error correction justifying Config C exploration.

---

## 10. Audit Artifacts & Data Preservation

- **Manifest Hash**: `3e6ad6f9ae2c08bbe0883d5c27475a8d8835888e2c82a7f5f53a0ea3d564f7b5`
- **Config A Results**: [`results/evaluation_heldout_100_config_a/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_a/inference_results.jsonl)
- **Config B Results**: [`results/evaluation_heldout_100_config_b/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_b/inference_results.jsonl)
- **Report Location**: [`reports/heldout_100_config_a_vs_b_results.md`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/reports/heldout_100_config_a_vs_b_results.md)
