import argparse
import sys
import os
from pathlib import Path
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.experiment_registry import ExperimentRegistry
from src.evaluation.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment ID (e.g. EXP-01)")
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke test mode")
    args = parser.parse_args()
    
    registry = ExperimentRegistry()
    try:
        run_meta = registry.initialize_run(args.exp)
    except ValueError as e:
        print(e)
        sys.exit(1)
        
    print(f"Initialized {args.exp} -> {run_meta['run_id']}")
    
    exp_type = run_meta["type"]
    base_config = run_meta["base_config"]
    
    if exp_type == "classical":
        print(f"Launching Classical Baseline Pipeline for {args.exp}")
        cmd = [sys.executable, "scripts/train_multimodal_baseline.py", "--config", base_config, "--run-dir", run_meta["run_dir"]]
        if args.smoke_test:
            cmd.append("--smoke-test")
            
        try:
            subprocess.run(cmd, check=True)
            eval_cmd = [sys.executable, "scripts/evaluate_multimodal_baseline.py", "--config", base_config, "--run-dir", run_meta["run_dir"]]
            if args.smoke_test:
                eval_cmd.append("--smoke-test")
            subprocess.run(eval_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Experiment pipeline failed or dependencies missing.")
            registry.update_status(run_meta["run_dir"], "FAILED")
            sys.exit(1)
            
    elif exp_type == "vlm":
        print(f"Launching VLM Pipeline for {args.exp} (Mode: {run_meta.get('cultural_mode')})")
        cmd = [sys.executable, "scripts/run_vlm_evaluation.py", "--config", base_config, "--run-dir", run_meta["run_dir"]]
        if args.smoke_test:
            cmd.append("--smoke-test")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("VLM pipeline failed or dependencies missing.")
            registry.update_status(run_meta["run_dir"], "FAILED")
            sys.exit(1)
            
    print("Generating Reports...")
    reporter = ReportGenerator(run_meta["run_dir"])
    
    try:
        reporter.generate_report(str(Path(run_meta["run_dir"]) / "raw_predictions" / "predictions.csv"), exp_type)
        registry.update_status(run_meta["run_dir"], "SUCCESS")
    except Exception as e:
        print(f"Report generation failed: {e}")
        registry.update_status(run_meta["run_dir"], "FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
