import argparse
import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.baseline.classifier import MultimodalClassifier
from src.baseline.trainer import MemotionDataset, train_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/multimodal_baseline.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny training loop")
    parser.add_argument("--run-dir", type=str, help="Override results directory")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.run_dir:
        config["training"]["results_dir"] = args.run_dir
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    try:
        from transformers import ViTImageProcessor, XLMRobertaTokenizer
        processor = ViTImageProcessor.from_pretrained(config["models"]["image_encoder"])
        tokenizer = XLMRobertaTokenizer.from_pretrained(config["models"]["text_encoder"])
    except Exception as e:
        print(f"Skipping baseline training: dependencies not satisfied ({e})")
        sys.exit(0)
        
    print("Loading datasets...")
    train_dataset = MemotionDataset(config["dataset"]["train_csv"], config["dataset"]["image_dir"], processor, tokenizer)
    val_dataset = MemotionDataset(config["dataset"]["val_csv"], config["dataset"]["image_dir"], processor, tokenizer)
    
    if args.smoke_test:
        print("Running smoke test mode (10 samples, 1 epoch)...")
        train_dataset.df = train_dataset.df.head(10)
        val_dataset.df = val_dataset.df.head(10)
        config["training"]["epochs"] = 1
        
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])
    
    print("Initializing Model...")
    model = MultimodalClassifier(
        image_model_id=config["models"]["image_encoder"],
        text_model_id=config["models"]["text_encoder"],
        hidden_size=config["model_config"]["hidden_size"],
        dropout=config["model_config"]["dropout"],
        mode=config["model_config"]["mode"]
    )
    
    print("Starting Training...")
    train_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        lr=float(config["training"]["learning_rate"]),
        device=device,
        save_dir=config["training"]["results_dir"]
    )

if __name__ == "__main__":
    main()
