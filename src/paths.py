"""
Centralne putanje projekta — da se nijedna ne ponavlja po skriptama.

Putanje se mogu preusmeriti preko promenljivih okruzenja (koristi se u
smoke testu nad sintetickim datasetom, da se ne dira pravi dataset):
    KOSNICA_ML_DATA_RAW, KOSNICA_ML_SPLITS, KOSNICA_ML_RUNS, KOSNICA_ML_MODELS
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


DATA_RAW = _dir("KOSNICA_ML_DATA_RAW", ROOT / "data" / "raw")
DATA_SPLITS = _dir("KOSNICA_ML_SPLITS", ROOT / "data" / "splits")
RUNS = _dir("KOSNICA_ML_RUNS", ROOT / "runs")
MODELS = _dir("KOSNICA_ML_MODELS", ROOT / "models")
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"

# Struktura BeeImage dataseta nakon raspakivanja
BEE_IMAGES_DIR = DATA_RAW / "bee_imgs"
BEE_CSV = DATA_RAW / "bee_data.csv"

KAGGLE_DATASET = "jenny18/honey-bee-annotated-images"
