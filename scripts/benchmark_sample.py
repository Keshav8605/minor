import time
import json
import logging
from src.vlm.inference import VLMInferenceEngine
from src.cultural.context_builder import CulturalContextBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_targeted_benchmark():
    engine = VLMInferenceEngine("configs/model.yaml")
    engine.load()
    
    builder = CulturalContextBuilder(
        categories_path="data/cultural/cultural_categories.json",
        knowledge_path="data/cultural/cultural_knowledge.json",
        drishtikon_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
        drishtikon_enabled=True,
        drishtikon_top_k=3,
        min_similarity_threshold=0.15
    )
    
    ocr_text = "आप फौजी होंगे IPL suspended for this season Meanwhile IPL fans : Ab zinda rehne ke liye bacha hi kya hai?"
    
    cult_context, cult_sources, drish_matches = builder.build_context(ocr_text)
    
    print("--- Cultural Context ---")
    print(cult_context)
    print("--- Matches ---")
    print(len(drish_matches))
    
    start_time = time.time()
    res = engine.infer(
        "data/processed/images/train_2837.jpg",
        mode="cultural",
        ocr_text=ocr_text,
        retrieved_context=cult_context,
        cultural_sources=cult_sources
    )
    end_time = time.time()
    
    print("\n--- Benchmark Result ---")
    print(json.dumps(res, indent=2))
    print(f"\nTotal Inference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_targeted_benchmark()
