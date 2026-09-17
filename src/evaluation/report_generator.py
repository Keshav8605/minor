import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ReportGenerator:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.tables_dir = self.run_dir / "tables"
        self.figures_dir = self.run_dir / "figures"
        self.reports_dir = self.run_dir / "reports"
        
    def generate_report(self, predictions_csv: str, experiment_type: str):
        if not Path(predictions_csv).exists():
            print(f"Cannot generate report, {predictions_csv} does not exist.")
            return
            
        df = pd.read_csv(predictions_csv)
        total_samples = len(df)
        
        if "prediction" in df.columns and "ground_truth" in df.columns:
            valid_df = df.dropna(subset=["prediction", "ground_truth"])
            y_true = valid_df["ground_truth"].astype(int)
            y_pred = valid_df["prediction"].astype(int)
            
            successful_preds = len(valid_df)
            failed_preds = total_samples - successful_preds
            
            metrics = {
                "total_samples": total_samples,
                "successful_predictions": successful_preds,
                "failed_predictions": failed_preds,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            }
        else:
            metrics = {}
            
        if experiment_type == "vlm" and "parsing_status" in df.columns:
            parsing_failures = len(df[df["parsing_status"] == "FAILED"])
            metrics["parsing_failure_rate"] = parsing_failures / max(1, total_samples)
            
        if experiment_type == "vlm" and "error_status" in df.columns:
            inference_failures = len(df[df["error_status"].str.contains("INFERENCE_ERROR", na=False)])
            metrics["inference_failure_rate"] = inference_failures / max(1, total_samples)
            
        if "cultural_dependency" in df.columns:
            grouped = df.dropna(subset=["prediction", "ground_truth"]).groupby("cultural_dependency")
            subgroup_metrics = {}
            for name, group in grouped:
                if len(group) > 0:
                    sub_y_true = group["ground_truth"].astype(int)
                    sub_y_pred = group["prediction"].astype(int)
                    subgroup_metrics[name] = {
                        "n": len(group),
                        "macro_f1": f1_score(sub_y_true, sub_y_pred, average="macro", zero_division=0)
                    }
            metrics["cultural_dependency_analysis"] = subgroup_metrics
            
        with open(self.reports_dir / "final_report.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if "confusion_matrix" in metrics:
                plt.figure(figsize=(6,5))
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.savefig(self.figures_dir / "confusion_matrix.png")
                plt.close()
                
        except ImportError:
            print("Matplotlib/Seaborn not installed, skipping plot generation.")
            
        metrics_df = pd.DataFrame([metrics])
        for col in ["confusion_matrix", "cultural_dependency_analysis"]:
            if col in metrics_df.columns:
                metrics_df = metrics_df.drop(columns=[col])
        metrics_df.to_csv(self.tables_dir / "metrics_summary.csv", index=False)
        return metrics

    def generate_comparison_report(self, general_csv: str, cultural_csv: str):
        """
        Generates a side-by-side comparative report of General vs Cultural-Aware VLM modes.
        """
        if not Path(general_csv).exists() or not Path(cultural_csv).exists():
            print(f"Missing predictions file: general={general_csv}, cultural={cultural_csv}")
            return None

        df_gen = pd.read_csv(general_csv)
        df_cul = pd.read_csv(cultural_csv)

        def get_eval_metrics(df):
            valid = df.dropna(subset=["prediction", "ground_truth"])
            if len(valid) == 0:
                return {}
            y_t = valid["ground_truth"].astype(int)
            y_p = valid["prediction"].astype(int)
            return {
                "samples": len(valid),
                "accuracy": round(accuracy_score(y_t, y_p), 4),
                "precision": round(precision_score(y_t, y_p, zero_division=0), 4),
                "recall": round(recall_score(y_t, y_p, zero_division=0), 4),
                "macro_f1": round(f1_score(y_t, y_p, average="macro", zero_division=0), 4),
                "weighted_f1": round(f1_score(y_t, y_p, average="weighted", zero_division=0), 4),
            }

        gen_metrics = get_eval_metrics(df_gen)
        cul_metrics = get_eval_metrics(df_cul)

        comparison = {
            "general_mode": gen_metrics,
            "cultural_mode": cul_metrics,
            "delta": {
                k: round(cul_metrics.get(k, 0) - gen_metrics.get(k, 0), 4)
                for k in ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"]
                if k in gen_metrics and k in cul_metrics
            }
        }

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        with open(self.reports_dir / "comparison_report.json", "w") as f:
            json.dump(comparison, f, indent=2)

        comp_df = pd.DataFrame([
            {"Metric": k, "General Mode": gen_metrics.get(k, "N/A"), "Cultural-Aware": cul_metrics.get(k, "N/A"), "Delta": comparison["delta"].get(k, "N/A")}
            for k in ["accuracy", "precision", "recall", "macro_f1", "weighted_f1"]
        ])
        comp_df.to_csv(self.tables_dir / "general_vs_cultural_comparison.csv", index=False)
        return comparison

