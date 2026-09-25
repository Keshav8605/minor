import pandas as pd
import json

df = pd.read_csv("results/evaluation_1790249336/comparison_results.csv")

def evaluate_threshold(df, threshold, mode="general"):
    if mode == "general":
        y_true = df["ground_truth"].apply(lambda x: 1 if x == "Humorous" else 0)
        y_prob = df["general_prob"]
    else:
        y_true = df["ground_truth"].apply(lambda x: 1 if x == "Humorous" else 0)
        y_prob = df["cultural_prob"]

    y_pred = (y_prob >= threshold).astype(int)

    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2

    return {
        "threshold": threshold,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_humor": tp + fp,
        "n_non_humor": tn + fn
    }

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
results = []
for t in thresholds:
    res_gen = evaluate_threshold(df, t, "general")
    res_cult = evaluate_threshold(df, t, "cultural")
    results.append((t, res_gen, res_cult))

with open("results/evaluation_1790249336/analysis/threshold_analysis.md", "w") as f:
    f.write("# Exploratory threshold analysis on the controlled evaluation set.\\n\\n")
    for t, rg, rc in results:
        f.write(f"## Threshold {t}\\n")
        f.write("### General\\n")
        f.write(f"Acc: {rg['acc']:.2f}, F1: {rg['f1']:.2f}, Precision: {rg['precision']:.2f}, Recall: {rg['recall']:.2f}, Specificity: {rg['specificity']:.2f}\\n")
        f.write(f"TP: {rg['tp']}, TN: {rg['tn']}, FP: {rg['fp']}, FN: {rg['fn']}\\n")
        f.write("### Cultural\\n")
        f.write(f"Acc: {rc['acc']:.2f}, F1: {rc['f1']:.2f}, Precision: {rc['precision']:.2f}, Recall: {rc['recall']:.2f}, Specificity: {rc['specificity']:.2f}\\n")
        f.write(f"TP: {rc['tp']}, TN: {rc['tn']}, FP: {rc['fp']}, FN: {rc['fn']}\\n\\n")

print("Threshold analysis completed.")
