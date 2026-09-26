import json
import logging
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Load processor to get exact tokenizer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

# Resolve true/false token IDs
true_candidates = ["true", " true", "True", " True", "TRUE", " TRUE"]
false_candidates = ["false", " false", "False", " False", "FALSE", " FALSE"]

true_token_ids = set()
false_token_ids = set()

for c in true_candidates:
    ids = tokenizer.encode(c, add_special_tokens=False)
    if ids:
        true_token_ids.add(ids[0])
for c in false_candidates:
    ids = tokenizer.encode(c, add_special_tokens=False)
    if ids:
        false_token_ids.add(ids[0])

def test_extraction(json_str):
    logging.info(f"\\n--- Testing Output ---\\n{json_str}")
    tokens = tokenizer.encode(json_str, add_special_tokens=False)
    
    # Simulate generated_ids_trimmed
    generated_token_list = tokens
    
    # The extraction logic
    humor_token_pos = None
    for pos in range(len(generated_token_list)-1, -1, -1):
        token_id = generated_token_list[pos]
        if token_id in true_token_ids or token_id in false_token_ids:
            humor_token_pos = pos
            break
            
    if humor_token_pos is not None:
        matched_token_text = tokenizer.decode([generated_token_list[humor_token_pos]])
        logging.info(f"Selected Token Index: {humor_token_pos}")
        logging.info(f"Selected Token Text: '{matched_token_text}'")
        logging.info(f"Is it the last 'true'/'false' in the sequence? Yes.")
        
        # Prove earlier occurrences are bypassed
        all_matches = []
        for i, tid in enumerate(generated_token_list):
            if tid in true_token_ids or tid in false_token_ids:
                all_matches.append((i, tokenizer.decode([tid])))
        logging.info(f"All occurrences in sequence: {all_matches}")
    else:
        logging.info("Failed to find true/false token")


mock_1 = """{
  "detected_text": "text",
  "visual_description": "image",
  "humor_evidence": "It is true that it is funny.",
  "non_humor_evidence": "none",
  "reason": "This is false advertising, which is true...",
  "humorous": true
}"""

mock_2 = """{
  "detected_text": "text",
  "visual_description": "image",
  "cultural_category": "cricket",
  "cultural_dependency": "low",
  "cultural_context": "None",
  "humor_evidence": "The claim is false.",
  "non_humor_evidence": "It is true that there is no humor.",
  "reason": "True humor is missing.",
  "humorous": false
}"""

test_extraction(mock_1)
test_extraction(mock_2)
