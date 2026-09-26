import os
import json
import csv
import statistics

eval_dir = "results/evaluation_1790249336"
analysis_dir = os.path.join(eval_dir, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

# Load raw records
jsonl_path = os.path.join(eval_dir, "inference_results.jsonl")
records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Deduplicate records by (sample_index, condition) taking the LAST occurrence
# (in case the robust wrapper retried something)
unique_records_map = {}
for r in records:
    unique_records_map[(r["sample_index"], r["condition"])] = r

final_records = list(unique_records_map.values())

# Group by sample_index
samples = {}
for r in final_records:
    s_idx = r["sample_index"]
    if s_idx not in samples:
        samples[s_idx] = {"general": None, "cultural": None}
    samples[s_idx][r["condition"]] = r

# Part 1 & 5: Metrics & Over-prediction
def calc_metrics(condition):
    tp = tn = fp = fn = 0
    probs = []
    latencies = []
    for s_idx, data in samples.items():
        r = data[condition]
        if not r: continue
        
        gt = r["ground_truth"]
        pred = r["prediction"]
        
        if r.get("humor_probability") is not None:
            probs.append(r["humor_probability"])
        if r.get("inference_duration_s") is not None:
            latencies.append(r["inference_duration_s"])
            
        if gt == 1 and pred == 1: tp += 1
        elif gt == 0 and pred == 0: tn += 1
        elif gt == 0 and pred == 1: fp += 1
        elif gt == 1 and pred == 0: fn += 1
        
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1,
        "Specificity": specificity,
        "Total_Predicted_Humorous": tp + fp,
        "Total_Predicted_Non_Humorous": tn + fn,
        "probs": probs,
        "latencies": latencies
    }

gen_m = calc_metrics("general")
cult_m = calc_metrics("cultural")

verified_metrics = {
    "General": {k: v for k, v in gen_m.items() if k not in ("probs", "latencies")},
    "Cultural-Aware": {k: v for k, v in cult_m.items() if k not in ("probs", "latencies")},
    "Deltas": {
        "Accuracy": cult_m["Accuracy"] - gen_m["Accuracy"],
        "F1": cult_m["F1"] - gen_m["F1"],
        "Recall": cult_m["Recall"] - gen_m["Recall"]
    }
}

with open(os.path.join(analysis_dir, "verified_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(verified_metrics, f, indent=2)

# Part 2 & 3: Prediction Changes
changes = []
improved_cnt = regressed_cnt = unch_corr = unch_incorr = 0

for s_idx, data in samples.items():
    g_rec = data.get("general")
    c_rec = data.get("cultural")
    
    if not g_rec or not c_rec: continue
    
    gt = g_rec["ground_truth"]
    g_pred = g_rec["prediction"]
    c_pred = c_rec["prediction"]
    
    g_prob = g_rec.get("humor_probability")
    c_prob = c_rec.get("humor_probability")
    
    case_type = ""
    if g_pred != gt and c_pred == gt:
        case_type = "Improved"
        improved_cnt += 1
    elif g_pred == gt and c_pred != gt:
        case_type = "Regressed"
        regressed_cnt += 1
    elif g_pred == gt and c_pred == gt:
        case_type = "Unchanged Correct"
        unch_corr += 1
    elif g_pred != gt and c_pred != gt:
        case_type = "Unchanged Incorrect"
        unch_incorr += 1
        
    delta = (c_prob - g_prob) if (c_prob is not None and g_prob is not None) else None
    
    changes.append({
        "sample_id": s_idx,
        "image_filename": g_rec["image_filename"],
        "ground_truth": gt,
        "general_prediction": g_pred,
        "cultural_prediction": c_pred,
        "general_probability": g_prob,
        "cultural_probability": c_prob,
        "probability_delta": delta,
        "ocr_text": g_rec.get("ocr_text", ""),
        "cultural_category": c_rec.get("cultural_category", ""),
        "cultural_dependency": c_rec.get("cultural_dependency", ""),
        "retrieved_context": c_rec.get("retrieved_context", ""),
        "drishtikon_matches": c_rec.get("drishtikon_matches_count", 0),
        "project_sources": len([s for s in c_rec.get("cultural_sources", []) if s != "DRISHTIKON"]),
        "case": case_type,
        "general_reasoning": g_rec.get("reasoning", ""),
        "cultural_reasoning": c_rec.get("reasoning", "")
    })

# Write prediction changes
with open(os.path.join(analysis_dir, "prediction_change_analysis.csv"), "w", newline="", encoding="utf-8") as f:
    if changes:
        writer = csv.DictWriter(f, fieldnames=changes[0].keys())
        writer.writeheader()
        writer.writerows(changes)

# Part 4: Retrieval Stats
retrieval_stats = {
    "DRISHTIKON_only": 0,
    "Project_only": 0,
    "Both": 0,
    "Neither": 0,
    "Directly_Related": "Subjective - pending manual review",
    "Changes_with_context": 0,
    "Changes_without_context": 0,
    "Zero_match_cases": 0
}

retrieval_rows = []
for c in changes:
    has_d = c["drishtikon_matches"] > 0
    has_p = c["project_sources"] > 0
    
    if has_d and has_p: retrieval_stats["Both"] += 1
    elif has_d: retrieval_stats["DRISHTIKON_only"] += 1
    elif has_p: retrieval_stats["Project_only"] += 1
    else: retrieval_stats["Neither"] += 1
    
    if not has_d and not has_p:
        retrieval_stats["Zero_match_cases"] += 1
        
    has_context = has_d or has_p
    is_changed = c["case"] in ["Improved", "Regressed"]
    
    if has_context and is_changed:
        retrieval_stats["Changes_with_context"] += 1
    elif not has_context and is_changed:
        retrieval_stats["Changes_without_context"] += 1
        
    retrieval_rows.append({
        "sample_id": c["sample_id"],
        "has_drishtikon": has_d,
        "has_project": has_p,
        "case": c["case"]
    })

with open(os.path.join(analysis_dir, "cultural_retrieval_analysis.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "has_drishtikon", "has_project", "case"])
    writer.writeheader()
    writer.writerows(retrieval_rows)

# Part 6: Probability Analysis
prob_deltas = [c["probability_delta"] for c in changes if c["probability_delta"] is not None]
prob_analysis = {
    "Mean_General_Prob": sum(gen_m["probs"])/len(gen_m["probs"]) if gen_m["probs"] else 0,
    "Mean_Cultural_Prob": sum(cult_m["probs"])/len(cult_m["probs"]) if cult_m["probs"] else 0,
    "Mean_Delta": sum(prob_deltas)/len(prob_deltas) if prob_deltas else 0,
    "Median_Delta": statistics.median(prob_deltas) if prob_deltas else 0,
    "Increases": len([d for d in prob_deltas if d > 0.01]),
    "Decreases": len([d for d in prob_deltas if d < -0.01]),
    "Unchanged": len([d for d in prob_deltas if abs(d) <= 0.01])
}

with open(os.path.join(analysis_dir, "probability_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(prob_analysis, f, indent=2)

# Part 7: Latency Analysis
lat_analysis = {
    "General": {
        "Mean": sum(gen_m["latencies"])/len(gen_m["latencies"]) if gen_m["latencies"] else 0,
        "Median": statistics.median(gen_m["latencies"]) if gen_m["latencies"] else 0,
        "Max": max(gen_m["latencies"]) if gen_m["latencies"] else 0,
        "Timeouts": len([r for r in final_records if r["condition"]=="general" and r.get("timed_out")])
    },
    "Cultural-Aware": {
        "Mean": sum(cult_m["latencies"])/len(cult_m["latencies"]) if cult_m["latencies"] else 0,
        "Median": statistics.median(cult_m["latencies"]) if cult_m["latencies"] else 0,
        "Max": max(cult_m["latencies"]) if cult_m["latencies"] else 0,
        "Timeouts": len([r for r in final_records if r["condition"]=="cultural" and r.get("timed_out")])
    },
    "Total_Inference_Time_s": sum(gen_m["latencies"]) + sum(cult_m["latencies"])
}

with open(os.path.join(analysis_dir, "latency_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(lat_analysis, f, indent=2)

# Part 8: Robustness
json_failures = len([r for r in final_records if r.get("json_error") is not None])
timeouts = len([r for r in final_records if r.get("timed_out")])
skipped = len([r for r in final_records if r.get("reasoning") and "SKIPPED" in r.get("reasoning")])

# Write Markdown reports
results_md = f"""# Results

## Overall Performance
The model exhibited a strong tendency toward the humorous class. In General mode, the accuracy was {verified_metrics['General']['Accuracy']:.2%} (F1: {verified_metrics['General']['F1']:.4f}). In Cultural-Aware mode, the accuracy was {verified_metrics['Cultural-Aware']['Accuracy']:.2%} (F1: {verified_metrics['Cultural-Aware']['F1']:.4f}). A modest improvement was observed, although these conclusions should be treated as preliminary due to the limited sample size of 20 unique memes.

## Confusion Matrix Analysis
**General Mode:**
- True Positives: {verified_metrics['General']['TP']}
- False Positives: {verified_metrics['General']['FP']}
- True Negatives: {verified_metrics['General']['TN']}
- False Negatives: {verified_metrics['General']['FN']}

**Cultural-Aware Mode:**
- True Positives: {verified_metrics['Cultural-Aware']['TP']}
- False Positives: {verified_metrics['Cultural-Aware']['FP']}
- True Negatives: {verified_metrics['Cultural-Aware']['TN']}
- False Negatives: {verified_metrics['Cultural-Aware']['FN']}

The true-negative rate (specificity) was low across both modes, indicating challenges in correctly rejecting non-humorous samples.

## Prediction Changes
On this evaluation sample:
- Improved (Incorrect -> Correct): {improved_cnt}
- Regressed (Correct -> Incorrect): {regressed_cnt}
- Unchanged Correct: {unch_corr}
- Unchanged Incorrect: {unch_incorr}

## Cultural Context Retrieval
- DRISHTIKON only: {retrieval_stats['DRISHTIKON_only']}
- Project only: {retrieval_stats['Project_only']}
- Both: {retrieval_stats['Both']}
- Neither (Zero match): {retrieval_stats['Neither']}

Changes associated with context: {retrieval_stats['Changes_with_context']}
Changes without context: {retrieval_stats['Changes_without_context']}

## Probability Analysis
The mean probability delta between the two modes was {prob_analysis['Mean_Delta']:.4f}. 
- Increases: {prob_analysis['Increases']}
- Decreases: {prob_analysis['Decreases']}
- Unchanged: {prob_analysis['Unchanged']}

*Note: Probability shifts must not be interpreted as improved predictions unless binary classification also improved.*

## Inference Latency
Cultural-Aware inference may require more computation because the prompt can contain retrieved context.
- General Mean: {lat_analysis['General']['Mean']:.2f}s (Max: {lat_analysis['General']['Max']}s)
- Cultural Mean: {lat_analysis['Cultural-Aware']['Mean']:.2f}s (Max: {lat_analysis['Cultural-Aware']['Max']}s)

## Evaluation Limitations
The results provide initial empirical support, but the experiment does not establish causality. The evaluation dataset contains only 20 unique memes. This behavior warrants further evaluation with larger sample sizes.

## Final Research Conclusion
Did adding retrieved Indian cultural context improve performance in this controlled experiment?
The measured result indicates an increase in Accuracy ({verified_metrics['General']['Accuracy']:.2%} -> {verified_metrics['Cultural-Aware']['Accuracy']:.2%}) and F1 score ({verified_metrics['General']['F1']:.4f} -> {verified_metrics['Cultural-Aware']['F1']:.4f}). However, the improvement is modest. Recall increased substantially, but false positives remained very high. Some predictions improved while others regressed. Therefore, the result is preliminary rather than definitive.
"""

with open(os.path.join(analysis_dir, "results_discussion.md"), "w", encoding="utf-8") as f:
    f.write(results_md)

tables_md = f"""# Evaluation Tables

## Helped vs Harmed Cases

| Case | Ground Truth | General | Cultural-Aware | Effect | Relevant Cultural Context |
|------|--------------|---------|----------------|--------|----------------------------|
"""
for c in changes:
    tables_md += f"| {c['sample_id']} | {c['ground_truth']} | {c['general_prediction']} | {c['cultural_prediction']} | {c['case']} | {len(c['retrieved_context']) > 0} |\n"

with open(os.path.join(analysis_dir, "evaluation_tables.md"), "w", encoding="utf-8") as f:
    f.write(tables_md)

# Print terminal summary
print("===== VERIFIED GENERAL METRICS =====")
print(json.dumps(verified_metrics["General"], indent=2))
print("===== VERIFIED CULTURAL METRICS =====")
print(json.dumps(verified_metrics["Cultural-Aware"], indent=2))
print("===== METRIC DELTAS =====")
print(json.dumps(verified_metrics["Deltas"], indent=2))
print(f"Improved: {improved_cnt}, Regressed: {regressed_cnt}")
print(f"Unchanged Correct: {unch_corr}, Unchanged Incorrect: {unch_incorr}")
print("===== RETRIEVAL STATS =====")
print(json.dumps(retrieval_stats, indent=2))
print(f"Timeouts: {timeouts}, JSON Failures: {json_failures}, Skipped/Recovered: {skipped}")
print("===== LATENCY STATS =====")
print(json.dumps(lat_analysis, indent=2))
print("===== ARTIFACT PATHS =====")
for fname in os.listdir(analysis_dir):
    print(os.path.abspath(os.path.join(analysis_dir, fname)))
