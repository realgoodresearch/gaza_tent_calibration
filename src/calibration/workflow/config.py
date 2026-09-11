from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_PATH = SRC_ROOT / "validation" / "prediction_accuracy_results.csv"
ARTIFACTS_DIR = SRC_ROOT / "calibration" / "artifacts"
