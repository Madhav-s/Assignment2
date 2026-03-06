from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PRED_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"

for p in (MODELS_DIR, PRED_DIR, METRICS_DIR):
    p.mkdir(parents=True, exist_ok=True)
