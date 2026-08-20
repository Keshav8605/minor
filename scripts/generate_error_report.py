import argparse
import sys
import os
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.error_analyzer import ErrorAnalyzer
from src.analysis.sample_selector import SampleSelector
from src.analysis.category_analyzer import CategoryAnalyzer

def create_html_report(df_high_conf, output_path):
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .sample { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .img-container { max-width: 300px; }
            img { max-width: 100%; height: auto; }
            .metadata { background: #f9f9f9; padding: 10px; margin-top: 10px; }
            .highlight { color: #d9534f; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Error Analysis Report</h1>
        <h2>High Confidence Errors</h2>
        <p>These samples were predicted incorrectly but with high confidence (>0.8). They are critical for identifying systematic model biases or label ambiguity.</p>
    """
    
    for idx, row in df_high_conf.iterrows():
        html += f"""
        <div class="sample">
            <h3>Sample ID: {row['sample_id']}</h3>
            <div class="img-container">
                <img src="file:///{os.path.abspath(str(row['image_path']))}" alt="Meme Image">
            </div>
            <div class="metadata">
                <p><strong>Ground Truth:</strong> {row['ground_truth']}</p>
                <p class="highlight"><strong>Prediction:</strong> {row['prediction']} (Confidence: {row['confidence']})</p>
                <p><strong>Raw Response:</strong> {row.get('raw_response', 'N/A')}</p>
                <p><strong>Auto-Inferred Reason:</strong> HYPOTHESIS REQUIRES HUMAN REVIEW</p>
            </div>
        </div>
        """
        
    html += "</body></html>"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the experiment run directory (e.g. results/EXP-04/...)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    predictions_csv = run_dir / "raw_predictions" / "predictions.csv"
    error_out_dir = run_dir / "error_analysis"
    
    if not predictions_csv.exists():
        print(f"Predictions file not found at {predictions_csv}")
        sys.exit(1)
        
    print("Generating Human Review Worksheet...")
    analyzer = ErrorAnalyzer(str(predictions_csv), str(error_out_dir))
    worksheet_path = analyzer.generate_review_worksheet()
    
    if worksheet_path is None:
        print("Exiting.")
        sys.exit(0)
        
    df = pd.read_csv(worksheet_path)
    selector = SampleSelector(df)
    
    high_conf_errors = selector.get_high_confidence_errors(threshold=0.8)
    
    html_path = error_out_dir / "error_report.html"
    create_html_report(high_conf_errors, html_path)
    print(f"Generated HTML report highlighting high-confidence errors at {html_path}")

if __name__ == "__main__":
    main()
