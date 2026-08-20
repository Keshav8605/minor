from sklearn.model_selection import train_test_split
import pandas as pd

def split_train_val(df: pd.DataFrame, target_col: str, val_size: float = 0.1, random_state: int = 42):
    """Splits dataset into train and validation with stratification."""
    train_df, val_df = train_test_split(
        df, 
        test_size=val_size, 
        stratify=df[target_col], 
        random_state=random_state
    )
    return train_df, val_df
