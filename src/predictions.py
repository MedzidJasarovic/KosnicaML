"""
Verovatnoće modela po slici — računaju se jednom i čuvaju, da notebook-ovi i
izveštaj (ROC krive, analiza grešaka) ne bi svaki put ponovo propuštali hiljade
slika kroz mrežu.

    runs/varroa_<arh>/pred_test.csv        test skup VarroaDataset-a
    runs/varroa_<arh>/pred_beedataset.csv  BeeDataset (nezavisan test)

Kolone: path, true (0 = zdrava, 1 = varoa), p_varoa, pred (prag 0,5).

    python -m src.predictions            # svi varroa_* runovi
"""
from __future__ import annotations

import sys

import pandas as pd
import torch
from PIL import Image

from .dataset import load_split
from .evaluate import load_run
from .field_test import beedataset
from .models import build_model, pick_device
from .paths import RUNS

BATCH = 64


def _model(run: str):
    config = load_run(run)
    model, _, eval_tf = build_model(config["arch"], 2, "letterbox", config.get("img_size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    return model.eval().to(pick_device()), eval_tf


@torch.no_grad()
def _score(model, eval_tf, paths) -> list[float]:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(paths), BATCH):
        batch = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])
        out += torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().tolist()
    return out


def predictions(run: str, which: str = "test", refresh: bool = False) -> pd.DataFrame:
    """which: 'test' ili 'beedataset'. Rezultat se kešira u runs/<run>/pred_<which>.csv."""
    cache = RUNS / run / f"pred_{which}.csv"
    if cache.is_file() and not refresh:
        return pd.read_csv(cache)

    if which == "test":
        df = load_split("test", dataset="varroa")
        frame = pd.DataFrame({"path": df["path"].astype(str), "true": df["target"].astype(int),
                              "video": df["video"], "raw_label": df["raw_label"],
                              "boxes": df["boxes"].astype(str)})
    else:
        items = beedataset()
        frame = pd.DataFrame(items, columns=["path", "true"])

    model, eval_tf = _model(run)
    frame["p_varoa"] = _score(model, eval_tf, frame["path"].tolist())
    frame["pred"] = (frame["p_varoa"] >= 0.5).astype(int)
    frame.to_csv(cache, index=False)
    return frame


def main() -> int:
    runs = sorted(p.parent.name for p in RUNS.glob("varroa_*/config.json"))
    for run in runs:
        for which in ("test", "beedataset"):
            df = predictions(run, which, refresh=True)
            print(f"{run:28s} {which:11s} {len(df):5d} slika  tačnost {(df['true'] == df['pred']).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
