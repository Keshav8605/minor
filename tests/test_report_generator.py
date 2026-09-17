import unittest
import tempfile
import pandas as pd
from pathlib import Path
from src.evaluation.report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        self.reporter = ReportGenerator(str(self.run_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_comparison_report(self):
        # Create dummy general and cultural prediction CSVs
        gen_df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "ground_truth": [1, 0, 1, 0],
            "prediction": [1, 1, 1, 0],
            "confidence": [0.8, 0.7, 0.9, 0.6]
        })
        cul_df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "ground_truth": [1, 0, 1, 0],
            "prediction": [1, 0, 1, 0],
            "confidence": [0.85, 0.9, 0.92, 0.88]
        })
        gen_path = self.run_dir / "gen.csv"
        cul_path = self.run_dir / "cul.csv"
        gen_df.to_csv(gen_path, index=False)
        cul_df.to_csv(cul_path, index=False)

        comparison = self.reporter.generate_comparison_report(str(gen_path), str(cul_path))
        self.assertIsNotNone(comparison)
        self.assertIn("general_mode", comparison)
        self.assertIn("cultural_mode", comparison)
        self.assertIn("delta", comparison)
        self.assertEqual(comparison["general_mode"]["accuracy"], 0.75)
        self.assertEqual(comparison["cultural_mode"]["accuracy"], 1.0)
        self.assertEqual(comparison["delta"]["accuracy"], 0.25)
