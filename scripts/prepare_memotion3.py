import os
import json
import pandas as pd
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_preparation(data_csv_path, images_dir, output_dir, eval_memes_list=None):
    """
    Validates, cleans, and structures the Memotion 3 dataset for binary humor classification.
    """
    logging.info(f"Starting Memotion 3 dataset preparation...")
    
    if not os.path.exists(data_csv_path):
        logging.error(f"Dataset CSV not found at {data_csv_path}. Please download the Memotion 3 dataset.")
        return
        
    df = pd.read_csv(data_csv_path)
    logging.info(f"Loaded {len(df)} rows from {data_csv_path}")
    
    # Inspect columns
    logging.info(f"Available columns: {list(df.columns)}")
    
    # Identify necessary columns (case-insensitive fallback)
    col_map = {c.lower(): c for c in df.columns}
    
    # Determine split column
    split_col = None
    if 'split' in col_map: split_col = col_map['split']
    
    # Determine image name column
    image_col = None
    for name in ['image_name', 'image', 'filename', 'file_name']:
        if name in col_map: 
            image_col = col_map[name]
            break
            
    # Determine text column
    text_col = None
    for name in ['text', 'ocr_text', 'ocr', 'caption']:
        if name in col_map:
            text_col = col_map[name]
            break
            
    # Determine humor column
    humor_col = None
    for name in ['humour', 'humor', 'humor_label']:
        if name in col_map:
            humor_col = col_map[name]
            break

    if not image_col or not humor_col or not text_col:
        logging.error("Could not find necessary columns (image, text, humor) in dataset.")
        return

    # Clean missing values
    df = df.dropna(subset=[image_col, humor_col, text_col])
    logging.info(f"Rows after dropping missing essential values: {len(df)}")
    
    # Ensure images actually exist
    valid_rows = []
    missing_images = 0
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, str(row[image_col]))
        if os.path.exists(img_path):
            valid_rows.append(row)
        else:
            missing_images += 1
            
    df = pd.DataFrame(valid_rows)
    logging.info(f"Missing image files: {missing_images}")
    logging.info(f"Rows with existing images: {len(df)}")
    
    # Prevent leakage: exclude the 20 eval memes
    eval_exclusion_count = 0
    if eval_memes_list:
        initial_len = len(df)
        df = df[~df[image_col].isin(eval_memes_list)]
        eval_exclusion_count = initial_len - len(df)
        logging.info(f"Excluded {eval_exclusion_count} images that belong to the controlled evaluation set.")

    # Label mapping
    # Memotion labels typically: "not_funny", "funny", "very_funny", "hilarious"
    def map_humor(label):
        lbl = str(label).lower().strip()
        if lbl in ["not_funny", "not funny", "non-humor", "0"]:
            return "NON-HUMOR"
        elif lbl in ["funny", "very_funny", "very funny", "hilarious", "humor", "1"]:
            return "HUMOR"
        return None
        
    df['binary_humor'] = df[humor_col].apply(map_humor)
    df = df.dropna(subset=['binary_humor'])
    logging.info(f"Rows after binary label mapping: {len(df)}")
    
    # Distribution
    dist = Counter(df['binary_humor'])
    logging.info(f"Class Distribution: {dist} ({dist['HUMOR']/len(df)*100:.1f}% HUMOR)")
    
    # Check for text duplicates
    text_dupes = df.duplicated(subset=[text_col]).sum()
    logging.info(f"Potential text duplicates detected: {text_dupes}")
    
    # Process splits
    if not split_col:
        logging.warning("No official split column found! Creating an 80/10/10 split.")
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['binary_humor'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['binary_humor'])
    else:
        train_df = df[df[split_col].astype(str).str.lower().isin(['train', 'training'])]
        val_df = df[df[split_col].astype(str).str.lower().isin(['val', 'validation', 'dev'])]
        test_df = df[df[split_col].astype(str).str.lower().isin(['test'])]
        
        logging.info(f"Official Split Counts - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    os.makedirs(output_dir, exist_ok=True)
    
    def save_jsonl(split_df, filename):
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                # Format for instruction tuning
                img_path = os.path.abspath(os.path.join(images_dir, str(row[image_col])))
                item = {
                    "image": img_path,
                    "ocr_text": str(row[text_col]),
                    "label": row['binary_humor'],
                    "original_label": str(row[humor_col]),
                    "messages": [
                        {"role": "system", "content": "You are a multimodal humor classification model. Output only HUMOR or NON-HUMOR."},
                        {"role": "user", "content": f"Analyze this Hindi/Hinglish meme using the image and text.\nText: {row[text_col]}\n\nDetermine whether it is humorous. Consider image, text, and their interaction."},
                        {"role": "assistant", "content": row['binary_humor']}
                    ]
                }
                f.write(json.dumps(item) + "\\n")
        logging.info(f"Saved {len(split_df)} records to {out_path}")

    save_jsonl(train_df, "memotion3_train.jsonl")
    save_jsonl(val_df, "memotion3_val.jsonl")
    save_jsonl(test_df, "memotion3_test.jsonl")
    
    logging.info("Preparation Complete.")

if __name__ == "__main__":
    # Dummy paths for local dry-run; adjust for actual execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/memotion3/labels.csv")
    parser.add_argument("--images", type=str, default="data/raw/memotion3/images")
    parser.add_argument("--out", type=str, default="data/processed/training")
    args = parser.parse_args()
    
    # We load the eval memes to prevent leakage
    eval_memes = []
    eval_csv = "results/evaluation_1790249336/comparison_results.csv"
    if os.path.exists(eval_csv):
        eval_df = pd.read_csv(eval_csv)
        eval_memes = eval_df['image_filename'].tolist()
        
    run_preparation(args.csv, args.images, args.out, eval_memes)
