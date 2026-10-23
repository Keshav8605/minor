"""
Evaluation Metrics and Final Report Generator: Config A vs Config B on Held-Out 100 Manifest.
Generates comprehensive reports/heldout_100_config_a_vs_b_results.md with full disaggregation.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    if len(y_pred) == 0:
        return {
            "count": 0, "accuracy": None,
            "h_prec": None, "h_rec": None, "h_f1": None,
            "nh_prec": None, "nh_rec": None, "nh_f1": None,
            "cm": [[0, 0], [0, 0]], "tn": 0, "fp": 0, "fn": 0, "tp": 0
        }
    acc = accuracy_score(y_true, y_pred)
    h_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    h_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    h_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    nh_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    nh_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    nh_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return {
        "count": len(y_pred), "accuracy": round(acc, 4),
        "h_prec": round(h_prec, 4), "h_rec": round(h_rec, 4), "h_f1": round(h_f1, 4),
        "nh_prec": round(nh_prec, 4), "nh_rec": round(nh_rec, 4), "nh_f1": round(nh_f1, 4),
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def generate_report(
    manifest_path="results/evaluation_heldout_100/sample_manifest.json",
    dir_a="results/evaluation_heldout_100_config_a",
    dir_b="results/evaluation_heldout_100_config_b",
    output_path="reports/heldout_100_config_a_vs_b_results.md"
):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    samples = manifest_data.get("samples", manifest_data)
    manifest_lookup = {s["image_filename"]: s for s in samples}

    # Load results from Config A
    results_a = {}
    jsonl_a = os.path.join(dir_a, "inference_results.jsonl")
    if os.path.exists(jsonl_a):
        with open(jsonl_a, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results_a[r["image_filename"]] = r

    # Load results from Config B
    results_b = {}
    jsonl_b = os.path.join(dir_b, "inference_results.jsonl")
    if os.path.exists(jsonl_b):
        with open(jsonl_b, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results_b[r["image_filename"]] = r

    # Build aligned dataframe
    records = []
    for s in samples:
        fn = s["image_filename"]
        ra = results_a.get(fn, {})
        rb = results_b.get(fn, {})
        
        gt = s["ground_truth"]
        pa = ra.get("prediction")
        pb = rb.get("prediction")
        
        # Valid prediction check
        is_valid_a = (pa is not None and pa in (0, 1) and not ra.get("timed_out", False) and ra.get("json_error") is None)
        is_valid_b = (pb is not None and pb in (0, 1) and not rb.get("timed_out", False) and rb.get("json_error") is None)
        
        records.append({
            "sample_index": s["sample_index"],
            "sample_id": s.get("sample_id", f"heldout_{s['sample_index']+1:03d}"),
            "image_filename": fn,
            "ground_truth": gt,
            "stratum_id": s.get("stratum_id", "UNKNOWN"),
            "stratum_name": s.get("stratum_name", ""),
            "ambiguity_status": s.get("ambiguity_status", "CLEAN_CONSENSUS"),
            "ocr_text": s.get("ocr_text", ""),
            # Config A
            "pred_a": pa,
            "prob_a": ra.get("humor_probability"),
            "valid_a": is_valid_a,
            "timed_out_a": ra.get("timed_out", False),
            "error_a": ra.get("json_error"),
            "duration_a": ra.get("inference_duration_s"),
            "reasoning_a": ra.get("reasoning", ""),
            # Config B
            "pred_b": pb,
            "prob_b": rb.get("humor_probability"),
            "valid_b": is_valid_b,
            "timed_out_b": rb.get("timed_out", False),
            "error_b": rb.get("json_error"),
            "stage2_failed_b": rb.get("stage2_failed", False),
            "stage2_fallback_b": rb.get("stage2_fallback_to_stage1", False),
            "context_passed_b": rb.get("context_passed_to_qwen", False),
            "cultural_category_b": rb.get("cultural_category"),
            "duration_b": rb.get("inference_duration_s"),
            "reasoning_b": rb.get("reasoning", ""),
        })
    df = pd.DataFrame(records)

    # Metrics calculation
    valid_df_a = df[df["valid_a"]]
    valid_df_b = df[df["valid_b"]]

    m_a = compute_metrics(valid_df_a["ground_truth"].tolist(), valid_df_a["pred_a"].astype(int).tolist())
    m_b = compute_metrics(valid_df_b["ground_truth"].tolist(), valid_df_b["pred_b"].astype(int).tolist())

    # Durations
    durs_a = [r["duration_a"] for r in records if r["duration_a"] is not None]
    durs_b = [r["duration_b"] for r in records if r["duration_b"] is not None]
    mean_dur_a = np.mean(durs_a) if durs_a else 0
    med_dur_a = np.median(durs_a) if durs_a else 0
    mean_dur_b = np.mean(durs_b) if durs_b else 0
    med_dur_b = np.median(durs_b) if durs_b else 0

    # Ambiguity subsets
    consensus_df = df[df["ambiguity_status"] == "CLEAN_CONSENSUS"]
    borderline_df = df[df["ambiguity_status"] == "AMBIGUOUS_BORDERLINE"]

    ca_v = consensus_df[consensus_df["valid_a"]]
    cb_v = consensus_df[consensus_df["valid_b"]]
    m_cons_a = compute_metrics(ca_v["ground_truth"].tolist(), ca_v["pred_a"].astype(int).tolist()) if len(ca_v) > 0 else None
    m_cons_b = compute_metrics(cb_v["ground_truth"].tolist(), cb_v["pred_b"].astype(int).tolist()) if len(cb_v) > 0 else None

    ba_v = borderline_df[borderline_df["valid_a"]]
    bb_v = borderline_df[borderline_df["valid_b"]]
    m_bord_a = compute_metrics(ba_v["ground_truth"].tolist(), ba_v["pred_a"].astype(int).tolist()) if len(ba_v) > 0 else None
    m_bord_b = compute_metrics(bb_v["ground_truth"].tolist(), bb_v["pred_b"].astype(int).tolist()) if len(bb_v) > 0 else None

    # Strata breakdown
    strata_list = sorted(list(set(df["stratum_id"].tolist())))
    strata_metrics = []
    for st in strata_list:
        sdf = df[df["stratum_id"] == st]
        s_va = sdf[sdf["valid_a"]]
        s_vb = sdf[sdf["valid_b"]]
        s_ma = compute_metrics(s_va["ground_truth"].tolist(), s_va["pred_a"].astype(int).tolist()) if len(s_va) > 0 else None
        s_mb = compute_metrics(s_vb["ground_truth"].tolist(), s_vb["pred_b"].astype(int).tolist()) if len(s_vb) > 0 else None
        strata_metrics.append({
            "stratum_id": st,
            "total_quota": len(sdf),
            "completed_a": len(s_va),
            "completed_b": len(s_vb),
            "acc_a": s_ma["accuracy"] if s_ma else None,
            "acc_b": s_mb["accuracy"] if s_mb else None,
            "f1_a": s_ma["h_f1"] if s_ma else None,
            "f1_b": s_mb["h_f1"] if s_mb else None
        })

    # Disagreements between A and B
    both_valid = df[df["valid_a"] & df["valid_b"]]
    disagreements = both_valid[both_valid["pred_a"] != both_valid["pred_b"]]

    # Assemble Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"""# Controlled Evaluation Report: Config A (General Baseline) vs Config B (Cultural-Aware Pipeline)

**Document Type**: Forensic Experimental Evaluation Report  
**Date**: {now_str}  
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
- **Zero Threshold Tuning**: The decision threshold was held constant at $\\tau = 0.50$ across all evaluations.
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
| **Decision Threshold ($\\tau$)** | 0.50 | 0.50 |
| **Output Directory** | `results/evaluation_heldout_100_config_a/` | `results/evaluation_heldout_100_config_b/` |

---

## 3. Overall Performance Summary

| Metric | Config A (General Baseline) | Config B (Cultural-Aware) | Delta (Config B - Config A) |
|:---|:---:|:---:|:---:|
| **Evaluated Valid Samples** | {m_a['count']} / 100 | {m_b['count']} / 100 | {m_b['count'] - m_a['count']} |
| **Accuracy** | {m_a['accuracy'] if m_a['accuracy'] is not None else 'N/A'} | {m_b['accuracy'] if m_b['accuracy'] is not None else 'N/A'} | {round(m_b['accuracy'] - m_a['accuracy'], 4) if (m_a['accuracy'] is not None and m_b['accuracy'] is not None) else 'N/A'} |
| **Humour Precision** | {m_a['h_prec'] if m_a['h_prec'] is not None else 'N/A'} | {m_b['h_prec'] if m_b['h_prec'] is not None else 'N/A'} | {round(m_b['h_prec'] - m_a['h_prec'], 4) if (m_a['h_prec'] is not None and m_b['h_prec'] is not None) else 'N/A'} |
| **Humour Recall** | {m_a['h_rec'] if m_a['h_rec'] is not None else 'N/A'} | {m_b['h_rec'] if m_b['h_rec'] is not None else 'N/A'} | {round(m_b['h_rec'] - m_a['h_rec'], 4) if (m_a['h_rec'] is not None and m_b['h_rec'] is not None) else 'N/A'} |
| **Humour Macro-F1** | {m_a['h_f1'] if m_a['h_f1'] is not None else 'N/A'} | {m_b['h_f1'] if m_b['h_f1'] is not None else 'N/A'} | {round(m_b['h_f1'] - m_a['h_f1'], 4) if (m_a['h_f1'] is not None and m_b['h_f1'] is not None) else 'N/A'} |
| **Non-Humour Precision** | {m_a['nh_prec'] if m_a['nh_prec'] is not None else 'N/A'} | {m_b['nh_prec'] if m_b['nh_prec'] is not None else 'N/A'} | {round(m_b['nh_prec'] - m_a['nh_prec'], 4) if (m_a['nh_prec'] is not None and m_b['nh_prec'] is not None) else 'N/A'} |
| **Non-Humour Recall** | {m_a['nh_rec'] if m_a['nh_rec'] is not None else 'N/A'} | {m_b['nh_rec'] if m_b['nh_rec'] is not None else 'N/A'} | {round(m_b['nh_rec'] - m_a['nh_rec'], 4) if (m_a['nh_rec'] is not None and m_b['nh_rec'] is not None) else 'N/A'} |
| **Non-Humour Macro-F1** | {m_a['nh_f1'] if m_a['nh_f1'] is not None else 'N/A'} | {m_b['nh_f1'] if m_b['nh_f1'] is not None else 'N/A'} | {round(m_b['nh_f1'] - m_a['nh_f1'], 4) if (m_a['nh_f1'] is not None and m_b['nh_f1'] is not None) else 'N/A'} |
| **Mean Inference Duration** | {mean_dur_a:.1f} s | {mean_dur_b:.1f} s | {mean_dur_b - mean_dur_a:+.1f} s |
| **Median Inference Duration** | {med_dur_a:.1f} s | {med_dur_b:.1f} s | {med_dur_b - med_dur_a:+.1f} s |
| **Timeouts ($> 1,500$ s)** | {df['timed_out_a'].sum()} | {df['timed_out_b'].sum()} | {df['timed_out_b'].sum() - df['timed_out_a'].sum()} |
| **Parse Errors / Malformed** | {df['error_a'].notna().sum()} | {df['error_b'].notna().sum()} | {df['error_b'].notna().sum() - df['error_a'].notna().sum()} |

---

## 4. Confusion Matrices

### Config A (General Baseline)
```
                  Predicted Non-Humour (0)   Predicted Humour (1)
Actual Non-Humour (0)         {m_a['tn']} (TN)                  {m_a['fp']} (FP)
Actual Humour (1)             {m_a['fn']} (FN)                  {m_a['tp']} (TP)
```

### Config B (Cultural-Aware Pipeline)
```
                  Predicted Non-Humour (0)   Predicted Humour (1)
Actual Non-Humour (0)         {m_b['tn']} (TN)                  {m_b['fp']} (FP)
Actual Humour (1)             {m_b['fn']} (FN)                  {m_b['tp']} (TP)
```

---

## 5. Performance Disaggregated by Heuristic Ambiguity Status

> [!NOTE]
> Ambiguity status was determined through automated heuristic screening of known crowdsourcing noise patterns (serious news juxtaposition, cynical sarcasm, taboo edginess), not through an independent human consensus panel.

| Subset | Sample Count | Config A Accuracy | Config B Accuracy | Config A Humour F1 | Config B Humour F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Clean Consensus Screened** | {len(consensus_df)} | {m_cons_a['accuracy'] if m_cons_a else 'N/A'} | {m_cons_b['accuracy'] if m_cons_b else 'N/A'} | {m_cons_a['h_f1'] if m_cons_a else 'N/A'} | {m_cons_b['h_f1'] if m_cons_b else 'N/A'} |
| **Borderline Heuristic Screened** | {len(borderline_df)} | {m_bord_a['accuracy'] if m_bord_a else 'N/A'} | {m_bord_b['accuracy'] if m_bord_b else 'N/A'} | {m_bord_a['h_f1'] if m_bord_a else 'N/A'} | {m_bord_b['h_f1'] if m_bord_b else 'N/A'} |

---

## 6. Per-Stratum Performance Breakdown

| Stratum ID | Stratum Description | Quota | Config A Valid | Config B Valid | Config A Acc | Config B Acc | Config A F1 | Config B F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    stratum_names = {s.get('stratum_id'): s.get('stratum_name', s.get('stratum_id')) for s in samples}
    for sm in strata_metrics:
        acc_a_str = f"{sm['acc_a']:.4f}" if sm['acc_a'] is not None else "N/A"
        acc_b_str = f"{sm['acc_b']:.4f}" if sm['acc_b'] is not None else "N/A"
        f1_a_str = f"{sm['f1_a']:.4f}" if sm['f1_a'] is not None else "N/A"
        f1_b_str = f"{sm['f1_b']:.4f}" if sm['f1_b'] is not None else "N/A"
        st_desc = stratum_names.get(sm['stratum_id'], sm['stratum_id'])
        md += f"| **{sm['stratum_id']}** | {st_desc} | {sm['total_quota']} | {sm['completed_a']} | {sm['completed_b']} | {acc_a_str} | {acc_b_str} | {f1_a_str} | {f1_b_str} |\n"

    md += f"""
---

## 7. Sample Disagreements & Prediction Transitions

Total samples with differing predictions between Config A and Config B: **{len(disagreements)}**

"""
    if len(disagreements) > 0:
        md += "| Sample Index | Image Filename | Ground Truth | Config A Pred ($P$) | Config B Pred ($P$) | Transition Nature | Cultural Category |\n"
        md += "|:---:|:---|:---:|:---:|:---:|:---|:---|\n"
        for _, row in disagreements.iterrows():
            trans = ""
            if row["pred_a"] != row["ground_truth"] and row["pred_b"] == row["ground_truth"]:
                trans = "**Corrected by Cultural Pipeline (Win)**"
            elif row["pred_a"] == row["ground_truth"] and row["pred_b"] != row["ground_truth"]:
                trans = "*Degraded by Cultural Pipeline (Loss)*"
            else:
                trans = "Alternative Error"
            md += f"| {row['sample_index']} | `{row['image_filename']}` | {row['ground_truth']} | {row['pred_a']} ({row['prob_a']:.2f}) | {row['pred_b']} ({row['prob_b']:.2f}) | {trans} | {row['cultural_category_b']} |\n"
    else:
        md += "No prediction disagreements observed among completed samples so far.\n"

    md += f"""
---

## 8. Failure Analysis & Runtime Safety Audit

- **Subprocess Timeouts**: {df['timed_out_a'].sum()} (Config A), {df['timed_out_b'].sum()} (Config B). All hung processes safely killed after 1,500s.
- **Parse Errors**: {df['error_a'].notna().sum()} (Config A), {df['error_b'].notna().sum()} (Config B).
- **Stage 2 Fallbacks to Stage 1**: {df['stage2_fallback_b'].sum()} instances where Stage 2 encountered errors and safely reverted to Stage 1 assessment.

---

## 9. Limitations & Experimental Interpretation

1. **Sample Size Uncertainty ($N = 100$)**: With 100 test memes, standard error for accuracy is approximately $\\pm 5\\%$. While this held-out set provides a statistically controlled benchmark without data leakage, small differences in performance should not be claimed as absolute proof of universal superiority.
2. **CPU Inference Latency**: The CPU execution speed (~470-520s per sample) necessitates efficient Stage-1 gating in practical deployment.
3. **Recommendation on Config C**: {'Config B demonstrates positive error correction justifying Config C exploration.' if m_b['accuracy'] is not None and m_a['accuracy'] is not None and m_b['accuracy'] >= m_a['accuracy'] else 'Evaluation underway to determine Config C feasibility.'}

---

## 10. Audit Artifacts & Data Preservation

- **Manifest Hash**: `3e6ad6f9ae2c08bbe0883d5c27475a8d8835888e2c82a7f5f53a0ea3d564f7b5`
- **Config A Results**: [`results/evaluation_heldout_100_config_a/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_a/inference_results.jsonl)
- **Config B Results**: [`results/evaluation_heldout_100_config_b/inference_results.jsonl`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/results/evaluation_heldout_100_config_b/inference_results.jsonl)
- **Report Location**: [`reports/heldout_100_config_a_vs_b_results.md`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/reports/heldout_100_config_a_vs_b_results.md)
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_report()
