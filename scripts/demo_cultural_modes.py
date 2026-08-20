import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cultural.context_retriever import CulturalRetriever
from src.cultural.cultural_prompt import build_prompt

def main():
    cat_path = "data/cultural/cultural_categories.json"
    kb_path = "data/cultural/cultural_knowledge.json"
    
    retriever = CulturalRetriever(cat_path, kb_path)
    
    # Synthetic Example
    image_path = "synthetic_meme.jpg"
    ocr_text = "When Sharma ji ka beta scores 99% in JEE but you just passed."
    
    print("=== SYNTHETIC MEME INPUT ===")
    print(f"Image: {image_path}")
    print(f"OCR: {ocr_text}\n")
    
    # Mode A: Plain
    print("=== MODE A: PLAIN VLM PROMPT ===")
    prompt_a = build_prompt(image_path, ocr_text, mode="A")
    print(json.dumps(prompt_a, indent=2))
    print("\n------------------------------------------------\n")
    
    # Mode B: Contextualized
    print("=== MODE B: CULTURALLY CONTEXTUALIZED PROMPT ===")
    retrieved_context = retriever.retrieve_context(ocr_text)
    print("-> Retrieved Context:")
    print(retrieved_context)
    print("\n-> Generated Prompt:")
    prompt_b = build_prompt(image_path, ocr_text, retrieved_context=retrieved_context, mode="B")
    print(json.dumps(prompt_b, indent=2))

if __name__ == "__main__":
    main()
