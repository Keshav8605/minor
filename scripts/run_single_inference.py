import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vlm.inference import VLMInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Run single VLM inference on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config")
    args = parser.parse_args()
    
    print("Initializing VLM Engine...")
    engine = VLMInferenceEngine(args.config)
    
    print(f"Loading model: {engine.model_id}...")
    try:
        engine.load()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    print(f"Model loaded successfully on {engine.device}.")
    print(f"Running inference on {args.image}...")
    
    try:
        result = engine.infer(args.image)
        print("\n=== INFERENCE RESULT ===")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
