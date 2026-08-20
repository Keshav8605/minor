import pandas as pd

class CategoryAnalyzer:
    @staticmethod
    def summarize_human_reviews(reviewed_csv_path: str):
        df = pd.read_csv(reviewed_csv_path)
        if "human_assigned_error_type" not in df.columns:
            return {}
            
        filled = df[df["human_assigned_error_type"].notna() & (df["human_assigned_error_type"] != "")]
        return filled["human_assigned_error_type"].value_counts().to_dict()
