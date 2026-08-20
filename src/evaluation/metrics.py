from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from pathlib import Path

def calculate_metrics(y_true, y_pred, output_dir: Path):
    """
    Calculates standard classification metrics and saves them to disk.
    """
    if not y_true or not y_pred:
        return {}
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics
