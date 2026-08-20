import pandas as pd

class SampleSelector:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_high_confidence_errors(self, threshold=0.8):
        if "confidence" not in self.df.columns:
            return pd.DataFrame()
        
        errors = self.df[self.df["prediction"] != self.df["ground_truth"]].copy()
        errors["confidence"] = pd.to_numeric(errors["confidence"], errors="coerce")
        return errors[errors["confidence"] >= threshold]
        
    def get_low_confidence_correct(self, threshold=0.6):
        if "confidence" not in self.df.columns:
            return pd.DataFrame()
            
        correct = self.df[self.df["prediction"] == self.df["ground_truth"]].copy()
        correct["confidence"] = pd.to_numeric(correct["confidence"], errors="coerce")
        return correct[correct["confidence"] <= threshold]
