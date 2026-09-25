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
        
        mode = self.experiment_config.get("mode", "general")
        context_builder = None
        if mode == "cultural":
            try:
                cultural_config_path = "configs/cultural.yaml"
                with open(cultural_config_path, "r") as f:
                    c_cfg = yaml.safe_load(f)
                from src.cultural.context_builder import CulturalContextBuilder
                context_builder = CulturalContextBuilder(
                    categories_path=c_cfg["categories_path"],
                    knowledge_path=c_cfg["knowledge_path"],
                    drishtikon_path=c_cfg.get("drishtikon_path"),
                    drishtikon_enabled=c_cfg.get("drishtikon_enabled", False),
                    drishtikon_top_k=c_cfg.get("drishtikon_top_k", 3),
                    min_similarity_threshold=c_cfg.get("min_similarity_threshold", 0.15)
                )
            except Exception as e:
                print(f"Warning: Could not initialize CulturalContextBuilder: {e}")

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
            cultural_category = ""
            cultural_dependency = ""
            cultural_context_used = False
            cultural_sources = []
            
            try:
                retrieved_context = ""
                ocr_text = str(row.get("ocr", row.get("text", row.get("OCR_text", ""))))
                if context_builder is not None and ocr_text:
                    retrieved_context, cultural_sources, _ = context_builder.build_context(ocr_text)

                result = self.engine.infer(
                    str(img_path), mode=mode,
                    ocr_text=ocr_text, retrieved_context=retrieved_context,
                    cultural_sources=cultural_sources
                )
                raw_response = result.get("raw_response", str(result))
                if result.get("error"):
                    parsing_status = "FAILED"
                else:
                    pred_bool = result.get("humorous")
                    prediction_val = 1 if pred_bool else 0
                    confidence = result.get("confidence")
                    cultural_category = result.get("cultural_category", "")
                    cultural_dependency = result.get("cultural_dependency", "")
                    cultural_context_used = bool(cultural_sources)
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
                error_status=error_status,
                cultural_category=cultural_category,
                cultural_dependency=cultural_dependency,
                cultural_context_used=cultural_context_used
            )
            
            if pd.notna(ground_truth) and prediction_val is not None:
                y_true.append(int(ground_truth))
                y_pred.append(prediction_val)
                
            print(f"Processed {sample_id} ({mode} mode): GT={ground_truth}, PRED={prediction_val}, CONF={confidence}")
            
        if y_true and y_pred:
            metrics_dir = Path(self.experiment_config["results_dir"]) / "metrics"
            calc = calculate_metrics(y_true, y_pred, metrics_dir)
            print("Evaluation Complete. Metrics:", calc)
        else:
            print("Evaluation Complete. (Check saved predictions for details)")
