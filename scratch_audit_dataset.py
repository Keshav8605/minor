import pandas as pd
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

for split in ['train', 'validation', 'test']:
    path = f'data/processed/{split}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{'='*60}")
        print(f"SPLIT: {split}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nHead (3 rows):\n{df.head(3).to_string()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nUnique values per column:")
        for c in df.columns:
            print(f"  {c}: {df[c].nunique()} unique")
        # Check for label distribution
        for c in df.columns:
            if df[c].nunique() < 20:
                print(f"\n  Value counts for '{c}':")
                print(f"  {df[c].value_counts().to_dict()}")
    else:
        print(f"MISSING: {path}")

# Check overlap between train/test
print("\n" + "="*60)
print("DATA LEAKAGE CHECK")
try:
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    val = pd.read_csv('data/processed/validation.csv')
    
    for col in ['image_path', 'image_name', 'id', 'image', 'image_filename', 'image_url', 'Unnamed: 0']:
        if col in train.columns and col in test.columns:
            overlap_tt = set(train[col].dropna()) & set(test[col].dropna())
            overlap_tv = set(train[col].dropna()) & set(val[col].dropna())
            overlap_vt = set(test[col].dropna()) & set(val[col].dropna())
            print(f"  Train-Test overlap on '{col}': {len(overlap_tt)} items")
            print(f"  Train-Val overlap on '{col}': {len(overlap_tv)} items")
            print(f"  Test-Val overlap on '{col}': {len(overlap_vt)} items")
except Exception as e:
    print(f"  Error: {e}")
