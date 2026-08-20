import pandas as pd
from pathlib import Path

ERROR_TAXONOMY = [
    "OCR/text extraction failure",
    "Hindi/Devanagari understanding failure",
    "Hinglish/code-mixing failure",
    "visual interpretation failure",
    "image-text relationship failure",
    "sarcasm",
    "cultural context failure",
    "ambiguous humor",
    "missing context",
    "label ambiguity",
    "other"
]

class ErrorAnalyzer:
    def __init__(self, predictions_csv: str, output_dir: str):
        self.predictions_csv = Path(predictions_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_review_worksheet(self):
        df = pd.read_csv(self.predictions_csv)
        
        if "prediction" not in df.columns or "ground_truth" not in df.columns:
            print("Missing required columns for error analysis.")
            return None
            
        valid = df.dropna(subset=["prediction", "ground_truth"])
        errors = valid[valid["prediction"].astype(int) != valid["ground_truth"].astype(int)].copy()
        
        if len(errors) == 0:
            print("No errors found to analyze!")
            return None
            
        errors["human_assigned_error_type"] = ""
        errors["auto_inferred_reason"] = "HYPOTHESIS REQUIRES HUMAN REVIEW"
        
        columns_to_keep = [
            "sample_id", "image_path", "ground_truth", "prediction", 
            "confidence", "human_assigned_error_type", "auto_inferred_reason", "raw_response"
        ]
        
        for col in ["cultural_category", "cultural_context_used"]:
            if col in errors.columns:
                columns_to_keep.append(col)
                
        columns_to_keep = [c for c in columns_to_keep if c in errors.columns]
        
        worksheet = errors[columns_to_keep]
        out_path = self.output_dir / "human_review_worksheet.csv"
        worksheet.to_csv(out_path, index=False)
        print(f"Generated human review worksheet with {len(errors)} samples at {out_path}")
        return out_path
