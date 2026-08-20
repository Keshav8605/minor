import yaml
import pandas as pd
from pathlib import Path

from src.evaluation.prediction_store import PredictionStore
from src.evaluation.metrics import calculate_metrics

class VLMEvaluator:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.dataset_config = self.config["dataset"]
        self.experiment_config = self.config["experiment"]
        self.model_config_path = self.config["model"]["config_path"]
        
        self.store = PredictionStore(self.experiment_config["results_dir"])
        self.engine = None
        
    def run(self):
        df = pd.read_csv(self.dataset_config["test_csv"])
        img_dir = Path(self.dataset_config["image_dir"])
        
        max_samples = self.experiment_config.get("max_samples", None)
        if max_samples:
            df = df.head(max_samples)
            
        completed_ids = self.store.load_completed_ids()
        
        try:
            from src.vlm.inference import VLMInferenceEngine
            self.engine = VLMInferenceEngine(self.model_config_path)
            self.engine.load()
        except Exception as e:
            print(f"Skipping evaluation: VLM could not be loaded ({e})")
            return
            
        y_true = []
        y_pred = []
        
        for idx, row in df.iterrows():
            sample_id = str(row.get("Unnamed: 0", idx))
            ground_truth = row.get("is_humorous", None)
            prediction_val = None
            
            if sample_id in completed_ids:
                print(f"Skipping {sample_id}, already processed.")
                continue
                
            img_path = img_dir / row["image_filename"]
            
            confidence = None
            raw_response = None
            parsing_status = "SUCCESS"
            error_status = "NONE"
            
            try:
                result = self.engine.infer(str(img_path))
                raw_response = result.get("raw_response", str(result))
                if result.get("error"):
                    parsing_status = "FAILED"
                else:
                    pred_bool = result.get("humorous")
                    prediction_val = 1 if pred_bool else 0
                    confidence = result.get("confidence")
            except Exception as e:
                error_status = f"INFERENCE_ERROR: {str(e)}"
                parsing_status = "FAILED"
                
            self.store.save_prediction(
                sample_id=sample_id,
                image_path=str(img_path),
                ground_truth=ground_truth,
                prediction=prediction_val,
                confidence=confidence,
                raw_response=raw_response,
                parsing_status=parsing_status,
                error_status=error_status
            )
            
            if pd.notna(ground_truth) and prediction_val is not None:
                y_true.append(int(ground_truth))
                y_pred.append(prediction_val)
                
            print(f"Processed {sample_id}: GT={ground_truth}, PRED={prediction_val}")
            
        if y_true and y_pred:
            metrics_dir = Path(self.experiment_config["results_dir"]) / "metrics"
            calc = calculate_metrics(y_true, y_pred, metrics_dir)
            print("Evaluation Complete. Metrics:", calc)
        else:
            print("Evaluation Complete. (Check saved predictions for details)")
