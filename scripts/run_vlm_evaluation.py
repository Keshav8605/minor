import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.evaluator import VLMEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run VLM evaluation")
    parser.add_argument("--config", type=str, default="configs/experiments/vlm_zero_shot.yaml", help="Path to experiment config")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny evaluation loop")
    parser.add_argument("--run-dir", type=str, help="Override results directory")
    args = parser.parse_args()
    
    print(f"Starting VLM Evaluation using config {args.config}")
    evaluator = VLMEvaluator(args.config)
    
    if args.run_dir:
        evaluator.experiment_config["results_dir"] = args.run_dir
        # Re-initialize PredictionStore and other paths with the new directory
        from src.evaluation.prediction_store import PredictionStore
        evaluator.prediction_store = PredictionStore(args.run_dir)
    
    if args.smoke_test:
        print("Running VLM evaluation in smoke test mode (2 samples)...")
        evaluator.experiment_config["max_samples"] = 2
        
    evaluator.run()

if __name__ == "__main__":
    main()
