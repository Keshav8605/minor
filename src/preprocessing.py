import pandas as pd

def normalize_humor_label(df: pd.DataFrame, source_col: str = "humour", target_col: str = "is_humorous", prefix: str = ""):
    """
    Creates a binary humor label from Memotion 3 raw annotations:
    not_funny -> 0
    funny, very_funny, hilarious -> 1
    """
    df = df.copy()
    
    def map_label(val):
        if val == "not_funny":
            return 0
        elif pd.isna(val):
            return None
        else:
            return 1
            
    if source_col in df.columns:
        df[target_col] = df[source_col].apply(map_label)
        df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(int)
    
    # Map index to image filename, preventing train/test collision by adding prefix
    if "Unnamed: 0" in df.columns:
        df["image_filename"] = prefix + df["Unnamed: 0"].astype(str) + ".jpg"
    
    return df
