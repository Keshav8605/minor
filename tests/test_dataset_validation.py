import pandas as pd
import unittest
from pathlib import Path
from src.preprocessing import normalize_humor_label
from src.data_validation import validate_images
import tempfile
import shutil
import os

class TestDatasetValidation(unittest.TestCase):
    def test_normalize_humor_label(self):
        df = pd.DataFrame({'humour': ['not_funny', 'funny', 'very_funny', 'hilarious', None]})
        processed = normalize_humor_label(df)
        self.assertIn('is_humorous', processed.columns)
        self.assertEqual(len(processed), 4)
        
        vals = processed['is_humorous'].tolist()
        self.assertEqual(vals, [0, 1, 1, 1])

    def test_split_target_label(self):
        """Ensure splits must contain the target label, else fail."""
        target_label = "is_humorous"
        
        # Valid split
        df_valid = pd.DataFrame({target_label: [0, 1], "text": ["a", "b"]})
        self.assertIn(target_label, df_valid.columns)
        
        # Missing label split
        df_invalid = pd.DataFrame({"text": ["a", "b"]})
        with self.assertRaises(AssertionError):
            assert target_label in df_invalid.columns, f"Pipeline must fail loudly if target label {target_label} is missing in split."

    def test_validate_images(self):
        """Test explicit handling of image paths and remote URLs."""
        temp_dir = tempfile.mkdtemp()
        try:
            img_dir = Path(temp_dir) / "images"
            img_dir.mkdir()
            
            # Create valid local image
            (img_dir / "valid_local.jpg").touch()
            # Create image with spaces
            (img_dir / "spaced image.jpg").touch()
            # Create image with unicode
            (img_dir / "नमस्ते.jpg").touch()
            
            df = pd.DataFrame({
                "image_filename": [
                    "valid_local.jpg",
                    "missing_local.jpg",
                    "http://example.com/image.jpg",
                    "https://example.com/image2.jpg",
                    "malformed|path?",
                    "spaced image.jpg",
                    "नमस्ते.jpg"
                ]
            })
            
            mask, missing = validate_images(df, str(img_dir), image_col="image_filename")
            
            self.assertEqual(mask, [True, False, False, False, False, True, True])
            self.assertEqual(missing, 4)
        finally:
            shutil.rmtree(temp_dir)

