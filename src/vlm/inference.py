import yaml
from pathlib import Path
from qwen_vl_utils import process_vision_info
from src.vlm.model_loader import load_model_and_processor
from src.vlm.prompts import build_humor_analysis_prompt
from src.vlm.output_parser import parse_json_response

class VLMInferenceEngine:
    def __init__(self, config_path: str = "configs/model.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.model_id = self.config["model_id"]
        self.generation_params = self.config.get("generation", {})
        
        self.model = None
        self.processor = None
        self.device = None
        
    def load(self):
        """Loads model and processor into memory."""
        self.model, self.processor, self.device = load_model_and_processor(self.model_id)
        
    def infer(self, image_path: str):
        """Runs a single inference on an image and returns parsed structured output."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
            
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")
            
        messages = build_humor_analysis_prompt(str(path))
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        generated_ids = self.model.generate(
            **inputs, 
            **self.generation_params
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return parse_json_response(output_text)
