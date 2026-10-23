"""
Master Orchestrator for Controlled Held-Out Evaluation: Config A vs Config B.
Runs sequentially with process isolation, outer 1,500s watchdog timeouts, atomic checkpoints,
and automatic final report generation.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

from scripts.generate_heldout_report import generate_report

MANIFEST_PATH = "results/evaluation_heldout_100/sample_manifest.json"
DIR_CONFIG_A = "results/evaluation_heldout_100_config_a"
DIR_CONFIG_B = "results/evaluation_heldout_100_config_b"

def run_config_watchdog(config_mode, results_dir, stage1_dir=None):
    print(f"\n========================================================")
    print(f"  LAUNCHING WATCHDOG: Config {config_mode.upper()} -> {results_dir}")
    print(f"========================================================\n")
    
    cmd = [
        "python", "-u", "run_robust_eval.py",
        "--manifest", MANIFEST_PATH,
        "--config-mode", config_mode,
        "--results-dir", results_dir
    ]
    if stage1_dir:
        cmd.extend(["--stage1-dir", stage1_dir])
        
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Watchdog exited with code {proc.returncode}")
        return False
    return True

def main():
    start_time = time.time()
    print(f"Held-out 100-Meme Evaluation Master Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Config A (General Baseline)
    print("\n--- PHASE 1: Executing Config A (General Baseline) ---")
    success_a = run_config_watchdog("general", DIR_CONFIG_A)
    if not success_a:
        print("Config A watchdog terminated with failure. Stopping before Config B.")
        sys.exit(1)
        
    # Phase 2: Config B (Cultural-Aware with Stage-1 Reuse)
    print("\n--- PHASE 2: Executing Config B (Cultural-Aware Two-Stage) ---")
    success_b = run_config_watchdog("cultural", DIR_CONFIG_B, stage1_dir=DIR_CONFIG_A)
    if not success_b:
        print("Config B watchdog terminated with failure.")
        sys.exit(1)
        
    # Phase 3: Final Metrics & Report Generation
    print("\n--- PHASE 3: Generating Final Held-Out Evaluation Report ---")
    generate_report(
        manifest_path=MANIFEST_PATH,
        dir_a=DIR_CONFIG_A,
        dir_b=DIR_CONFIG_B,
        output_path="reports/heldout_100_config_a_vs_b_results.md"
    )
    
    total_elapsed = time.time() - start_time
    print(f"\nEvaluation Pipeline Complete in {total_elapsed/3600:.2f} hours ({total_elapsed:.0f}s).")

if __name__ == "__main__":
    main()
