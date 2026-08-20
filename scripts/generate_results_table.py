import os
import json
import glob

def generate_results_table():
    results_dir = "results"
    output_file = "docs/research/RESULTS_TABLE.md"
    
    # Find all final_report.json files
    report_files = glob.glob(os.path.join(results_dir, "*", "*", "reports", "final_report.json"))
    
    table_lines = [
        "# Experimental Results\n",
        "This table is automatically generated from the execution logs to ensure zero metric fabrication.\n",
        "| Experiment ID | Macro F1 | Accuracy | Parsing Failure Rate |",
        "|---|---|---|---|"
    ]
    
    for report_file in sorted(report_files):
        # Extract experiment ID from path, assuming results/EXP-01/EXP-01_timestamp/reports/final_report.json
        parts = os.path.normpath(report_file).split(os.sep)
        if len(parts) >= 4:
            exp_id = parts[1]
        else:
            exp_id = "Unknown"
            
        with open(report_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                macro_f1 = data.get("macro_f1", 0.0)
                accuracy = data.get("accuracy", 0.0)
                
                # Calculate parsing failure rate
                successful = data.get("successful_predictions", 0)
                failed = data.get("failed_predictions", 0)
                total = successful + failed
                parsing_failure_rate = (failed / total) if total > 0 else 0.0
                
                table_lines.append(f"| {exp_id} | {macro_f1:.4f} | {accuracy:.4f} | {parsing_failure_rate:.2%} |")
            except Exception as e:
                print(f"Error reading {report_file}: {e}")
                
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(table_lines) + "\n")
        
    print(f"Results table generated successfully at {output_file}")

if __name__ == "__main__":
    generate_results_table()
