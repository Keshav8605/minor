import pandas as pd
from pathlib import Path

def validate_images(df: pd.DataFrame, image_dir: str, image_col: str = "image_filename"):
    """Checks if local images exist and returns a boolean mask of valid records."""
    img_dir_path = Path(image_dir)
    valid_mask = []
    missing_count = 0
    for img_name in df[image_col]:
        img_str = str(img_name)
        if img_str.startswith("http://") or img_str.startswith("https://"):
            # Remote URL, not a local file
            valid_mask.append(False)
            missing_count += 1
            continue
            
        if (img_dir_path / img_str).exists():
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            missing_count += 1
    return valid_mask, missing_count
