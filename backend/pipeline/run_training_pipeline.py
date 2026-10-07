"""
Master runner for the end-to-end training and evaluation pipeline:
1. download_data.py
2. curate_and_split.py
3. train_classical.py
4. train_gnn.py
5. evaluate_all.py
"""

import subprocess
import sys
from pathlib import Path

PYTHON_EXE = sys.executable
PIPELINE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("Download Datasets", PIPELINE_DIR / "download_data.py"),
    ("Curate & Split Data", PIPELINE_DIR / "curate_and_split.py"),
    ("Train Classical Model", PIPELINE_DIR / "train_classical.py"),
    ("Train Molecular GNN", PIPELINE_DIR / "train_gnn.py"),
    ("Evaluate All Models", PIPELINE_DIR / "evaluate_all.py")
]


def run_step(step_name: str, script_path: Path):
    print(f"\n{'='*70}\n[STEP] {step_name}\n{'='*70}")
    cmd = [PYTHON_EXE, str(script_path)]
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Step {step_name} failed with code {res.returncode}")


def main():
    print("STARTING COMPLETE hERG MODEL TRAINING PIPELINE")
    for name, script in SCRIPTS:
        run_step(name, script)
    print("\n[COMPLETE] Pipeline finished successfully! All artifacts generated.")


if __name__ == "__main__":
    main()
