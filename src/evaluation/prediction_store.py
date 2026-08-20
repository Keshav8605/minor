import csv
import threading
from pathlib import Path

class PredictionStore:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.predictions_file = self.results_dir / "predictions.csv"
        self._lock = threading.Lock()
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.predictions_file.exists():
            with open(self.predictions_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_id", 
                    "image_path", 
                    "ground_truth", 
                    "prediction", 
                    "confidence", 
                    "raw_response", 
                    "parsing_status", 
                    "error_status"
                ])
                
    def save_prediction(self, sample_id, image_path, ground_truth, prediction, confidence, raw_response, parsing_status, error_status):
        with self._lock:
            with open(self.predictions_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                safe_raw = str(raw_response).replace("\n", " ") if raw_response else ""
                writer.writerow([
                    sample_id,
                    image_path,
                    ground_truth,
                    prediction,
                    confidence,
                    safe_raw,
                    parsing_status,
                    error_status
                ])
            
    def load_completed_ids(self):
        if not self.predictions_file.exists():
            return set()
            
        completed = set()
        with open(self.predictions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["sample_id"])
        return completed
