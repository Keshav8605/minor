import pandas as pd
import os

print("=" * 60)
print("PHASE 1 & 2: MEMOTION 3 DATASET DEEP INSPECTION")
print("=" * 60)

for split_name, f in [("TRAIN", "train.csv"), ("VALIDATION", "validation.csv"), ("TEST", "test.csv")]:
    path = f"data/processed/{f}"
    df = pd.read_csv(path)
    print(f"\n--- {split_name} ({f}) ---")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Humor label values
    humour_vals = sorted(df["humour"].dropna().unique())
    print(f"  humour unique values: {humour_vals}")
    print(f"  humour distribution:")
    for val, cnt in df["humour"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {val}: {cnt} ({pct:.1f}%)")
    
    # is_humorous field
    if "is_humorous" in df.columns:
        ih_vals = sorted(df["is_humorous"].dropna().unique())
        print(f"  is_humorous unique: {ih_vals}")
        print(f"  is_humorous distribution:")
        for val, cnt in df["is_humorous"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {val}: {cnt} ({pct:.1f}%)")
    
    # Missing values
    print(f"  Missing humour: {df['humour'].isna().sum()}")
    print(f"  Missing ocr: {df['ocr'].isna().sum()}")
    print(f"  Missing image_filename: {df['image_filename'].isna().sum()}")
    
    # Sample values
    print(f"  Sample image_filename: {df['image_filename'].iloc[:3].tolist()}")
    print(f"  Sample ocr (first): {str(df['ocr'].iloc[0])[:100]}")

# Check images directory
img_dir = "data/processed/images"
if os.path.exists(img_dir):
    img_files = os.listdir(img_dir)
    print(f"\n--- IMAGES ---")
    print(f"  Total image files: {len(img_files)}")
    print(f"  Sample files: {img_files[:5]}")
    
    # Check how many CSV images actually exist
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    all_df = pd.concat([train_df, val_df, test_df])
    
    missing = 0
    found = 0
    for fname in all_df["image_filename"]:
        if os.path.exists(os.path.join(img_dir, str(fname))):
            found += 1
        else:
            missing += 1
    print(f"  Images found for CSV rows: {found}")
    print(f"  Images MISSING for CSV rows: {missing}")
else:
    print(f"\n  Images directory NOT found at {img_dir}")

# Check for duplicates across splits
print("\n--- DUPLICATE / LEAKAGE CHECK ---")
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_imgs = set(train_df["image_filename"].tolist())
val_imgs = set(val_df["image_filename"].tolist())
test_imgs = set(test_df["image_filename"].tolist())

tv_overlap = train_imgs & val_imgs
tt_overlap = train_imgs & test_imgs
vt_overlap = val_imgs & test_imgs

print(f"  Train-Val image overlap: {len(tv_overlap)}")
print(f"  Train-Test image overlap: {len(tt_overlap)}")
print(f"  Val-Test image overlap: {len(vt_overlap)}")

# Check against the 20 evaluation memes
eval_csv = "results/evaluation_1790249336/comparison_results.csv"
if os.path.exists(eval_csv):
    eval_df = pd.read_csv(eval_csv)
    if "image_filename" in eval_df.columns:
        eval_imgs = set(eval_df["image_filename"].tolist())
        print(f"\n--- EVALUATION MEME EXCLUSION CHECK ---")
        print(f"  Evaluation memes: {len(eval_imgs)}")
        print(f"  Eval filenames: {sorted(list(eval_imgs))[:10]}...")
        
        all_training_imgs = train_imgs | val_imgs | test_imgs
        overlap_with_eval = all_training_imgs & eval_imgs
        print(f"  Memotion images matching eval set: {len(overlap_with_eval)}")
        if overlap_with_eval:
            print(f"  OVERLAPPING FILES: {overlap_with_eval}")
    else:
        print(f"  Evaluation CSV columns: {list(eval_df.columns)}")
        print(f"  Cannot check overlap - 'image_filename' column not found")
else:
    print(f"  Evaluation CSV not found at {eval_csv}")
