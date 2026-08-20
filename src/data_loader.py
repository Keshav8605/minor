import pandas as pd
from pathlib import Path

def load_raw_data(raw_data_dir: str):
    """Loads raw dataset CSVs for Memotion 3."""
    base_dir = Path(raw_data_dir)
    train_csv = base_dir / "memotion3" / "memotion3" / "train.csv"
    test_csv = base_dir / "memotion3-test" / "test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Cannot find dataset CSV files in {base_dir}")
        
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    return df_train, df_test
