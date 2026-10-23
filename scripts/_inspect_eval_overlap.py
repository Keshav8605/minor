import pandas as pd

eval_df = pd.read_csv("results/evaluation_1790249336/comparison_results.csv")
eval_imgs = set(eval_df["image_filename"].tolist())

for split_name, f in [("TRAIN", "train.csv"), ("VALIDATION", "validation.csv"), ("TEST", "test.csv")]:
    df = pd.read_csv("data/processed/" + f)
    overlap = set(df["image_filename"].tolist()) & eval_imgs
    print(split_name + ": " + str(len(overlap)) + " eval memes found")
    if overlap:
        for img in sorted(overlap):
            row = df[df["image_filename"] == img].iloc[0]
            print("  " + img + " -> humour=" + str(row["humour"]) + ", is_humorous=" + str(row["is_humorous"]))

# Text duplicates
train_df = pd.read_csv("data/processed/train.csv")
text_dupes = train_df["ocr"].duplicated().sum()
print("\nTrain OCR text duplicates: " + str(text_dupes))

# Overall binary class distribution after eval exclusion
all_dfs = []
for f in ["train.csv", "validation.csv", "test.csv"]:
    df = pd.read_csv("data/processed/" + f)
    df["split"] = f.replace(".csv", "")
    all_dfs.append(df)

combined = pd.concat(all_dfs)
combined_excl = combined[~combined["image_filename"].isin(eval_imgs)]
print("\nAfter excluding 20 eval memes:")
print("Total: " + str(len(combined_excl)))
for split in ["train", "validation", "test"]:
    subset = combined_excl[combined_excl["split"] == split]
    h = (subset["is_humorous"] == 1).sum()
    nh = (subset["is_humorous"] == 0).sum()
    print("  " + split + ": " + str(len(subset)) + " (HUMOR=" + str(h) + ", NON-HUMOR=" + str(nh) + ")")
