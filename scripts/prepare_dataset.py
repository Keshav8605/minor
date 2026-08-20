import yaml
from pathlib import Path
import sys
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_raw_data
from src.data_validation import validate_images
from src.preprocessing import normalize_humor_label

def main():
    with open("configs/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = config["raw_data_dir"]
    out_dir = Path(config["processed_data_dir"])
    target_col = config["target_label"]
    
    df_train_raw, df_test_raw = load_raw_data(raw_dir)
    
    # Remove duplicates based on image_url
    df_train_raw = df_train_raw.drop_duplicates(subset=['image_url'])
    
    # Normalize labels and prepend 'train_' to filenames to prevent collision
    df_train_proc = normalize_humor_label(df_train_raw, target_col=target_col, prefix="train_")
    
    # Validate images exist in the original raw directory
    train_img_dir = Path(raw_dir) / "memotion3" / "memotion3" / "images"
    # Wait, the previous script had: train_img_dir = Path(raw_dir) / "trainImages" / "trainImages". Let's use the actual directory name if we know it. 
    # Let me check the directory first... I'll use the original logic but fix the image check.
    train_img_dir = Path(raw_dir) / "trainImages" / "trainImages" 
    
    # Actually, we need to map the filename back to the original without prefix to check existence
    df_train_proc["original_filename"] = df_train_proc["Unnamed: 0"].astype(str) + ".jpg"
    valid_mask, missing_train = validate_images(df_train_proc, train_img_dir, image_col="original_filename")
    df_train_proc = df_train_proc[valid_mask].copy()
    
    # Split train into train (80%), val (10%), test (10%)
    df_train, df_temp = train_test_split(df_train_proc, test_size=0.2, stratify=df_train_proc[target_col], random_state=config["random_seed"])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp[target_col], random_state=config["random_seed"])

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process blind test
    df_blind = normalize_humor_label(df_test_raw, target_col=target_col, prefix="blind_")
    df_blind["original_filename"] = df_blind["Unnamed: 0"].astype(str) + ".jpg"
    test_img_dir = Path(raw_dir) / "testImages" / "testImages"
    valid_test_mask, missing_test = validate_images(df_blind, test_img_dir, image_col="original_filename")
    df_blind = df_blind[valid_test_mask].copy()
    
    # Copy images to a unified directory so we don't have to juggle paths later
    unified_img_dir = out_dir / "images"
    unified_img_dir.mkdir(parents=True, exist_ok=True)
    
    print("Copying valid images to unified directory...")
    for _, row in pd.concat([df_train, df_val, df_test]).iterrows():
        src = train_img_dir / row["original_filename"]
        dst = unified_img_dir / row["image_filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
            
    for _, row in df_blind.iterrows():
        src = test_img_dir / row["original_filename"]
        dst = unified_img_dir / row["image_filename"]
        if not dst.exists():
            shutil.copy2(src, dst)

    df_train.drop(columns=["original_filename"]).to_csv(out_dir / "train.csv", index=False)
    df_val.drop(columns=["original_filename"]).to_csv(out_dir / "validation.csv", index=False)
    df_test.drop(columns=["original_filename"]).to_csv(out_dir / "test.csv", index=False)
    df_blind.drop(columns=["original_filename"]).to_csv(out_dir / "blind_test.csv", index=False)
    
    print("=== Dataset Summary ===")
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")
    print(f"Blind Test samples: {len(df_blind)}")

if __name__ == "__main__":
    main()
