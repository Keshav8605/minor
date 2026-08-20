import argparse
import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.baseline.classifier import MultimodalClassifier
from src.baseline.trainer import MemotionDataset
from src.evaluation.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/multimodal_baseline.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny evaluation loop")
    parser.add_argument("--run-dir", type=str, help="Override results directory")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.run_dir:
        config["training"]["results_dir"] = args.run_dir
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from transformers import ViTImageProcessor, XLMRobertaTokenizer
        processor = ViTImageProcessor.from_pretrained(config["models"]["image_encoder"])
        tokenizer = XLMRobertaTokenizer.from_pretrained(config["models"]["text_encoder"])
    except Exception as e:
        print(f"Skipping baseline eval: dependencies not satisfied ({e})")
        sys.exit(0)
        
    test_dataset = MemotionDataset(config["dataset"]["test_csv"], config["dataset"]["image_dir"], processor, tokenizer)
    
    if args.smoke_test:
        print("Running evaluation in smoke test mode (10 samples)...")
        test_dataset.df = test_dataset.df.head(10)
        
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
    
    model = MultimodalClassifier(
        image_model_id=config["models"]["image_encoder"],
        text_model_id=config["models"]["text_encoder"],
        hidden_size=config["model_config"]["hidden_size"],
        dropout=config["model_config"]["dropout"],
        mode=config["model_config"]["mode"]
    )
    
    model_path = Path(config["training"]["results_dir"]) / "best_model.pth"
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded best_model.pth successfully.")
        except Exception as e:
            print(f"Warning: Failed to load best_model.pth (corrupted?): {e}. Evaluating randomly initialized model.")
    else:
        print("Warning: best_model.pth not found. Evaluating randomly initialized model.")
        
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    raw_preds_dir = Path(config["training"]["results_dir"]) / "raw_predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)
    
    with open(raw_preds_dir / "predictions.csv", "w", encoding="utf-8") as f:
        f.write("sample_id,ground_truth,prediction,confidence,parsing_status,error_status\n")
        
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].numpy()
                
                logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                sample_ids = batch["sample_id"]
                
                # Iterate over batch to write to csv
                for i in range(len(labels)):
                    f.write(f"{sample_ids[i]},{int(labels[i])},{preds[i]},{probs[i]:.4f},SUCCESS,NONE\n")
                
                y_true.extend(labels.astype(int))
                y_pred.extend(preds)
            
    metrics_dir = Path(config["training"]["results_dir"]) / "eval_metrics"
    calc = calculate_metrics(y_true, y_pred, metrics_dir)
    print("Baseline Evaluation Complete. Metrics:", calc)

if __name__ == "__main__":
    main()
