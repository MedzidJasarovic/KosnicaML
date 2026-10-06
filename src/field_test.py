"""
Testovi van VarroaDataset-a — da li model radi na slikama snimljenim drugačije.

1. TERENSKE FOTOGRAFIJE (data/field_photos/{zdrava,varoa}/) — fotografije pojedinačnih
   pčela sa bilo kojom pozadinom. Ovo je slučaj korisnika aplikacije.
2. BeeDataset (Hickert, BeeAlarmed) — 7.490 slika sa druge kamere, zelena podloga,
   oznaka varoe po pčeli. Model ga nikad nije video: meri prenosivost na drugo snimanje.

Svi varroa_* runovi se testiraju istim slikama.

    python -m src.field_test
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from .evaluate import load_run
from .models import build_model, pick_device
from .paths import ROOT, RUNS

FIELD_DIR = ROOT / "data" / "field_photos"
BEEDATASET_DIR = ROOT / "data" / "beedataset"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
BATCH = 64


def _load_model(run: str):
    config = load_run(run)
    model, _, eval_tf = build_model(config["arch"], 2, "letterbox", config.get("img_size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    return config, model.eval().to(pick_device()), eval_tf


@torch.no_grad()
def _p_varoa(model, eval_tf, paths) -> np.ndarray:
    out = []
    for i in range(0, len(paths), BATCH):
        batch = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])
        device = next(model.parameters()).device
        out.append(torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def field_photos() -> list[tuple[str, int]]:
    items = []
    for target, folder in [(0, "zdrava"), (1, "varoa")]:
        for p in sorted((FIELD_DIR / folder).glob("*")):
            if p.suffix.lower() in IMAGE_EXT:
                items.append((str(p), target))
    return items


def beedataset() -> list[tuple[str, int]]:
    """Oznaka varroa iz data.json; slike 300x150 (visina x širina)."""
    meta = BEEDATASET_DIR / "data.json"
    if not meta.is_file():
        return []
    data = json.loads(meta.read_text())
    folder = BEEDATASET_DIR / "images_300"
    items = []
    for name, labels in data.items():
        path = folder / name
        if path.is_file():
            items.append((str(path), int(bool(labels.get("varroa", 0)))))
    return items


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {
        "slika": int(len(y)),
        "sa_varoom": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "recall_varoa": round(float(recall_score(y, pred, zero_division=0)), 4),
        "precision_varoa": round(float(precision_score(y, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="samo ovaj run (podrazumevano: svi završeni varroa_*)")
    args = parser.parse_args(argv)
    # samo završeni treninzi — config.json se piše na kraju treninga
    runs = [args.run] if args.run else sorted(p.parent.name for p in RUNS.glob("varroa_*/config.json"))
    if not runs:
        print("Nema istreniranih varroa_* modela.")
        return 1

    field = field_photos()
    bee = beedataset()
    print(f"Terenskih fotografija: {len(field)} | BeeDataset slika: {len(bee)}")

    results = {}
    for run in runs:
        config, model, eval_tf = _load_model(run)
        res = {"arch_display": config["arch_display"]}

        if field:
            paths, y = zip(*field)
            p = _p_varoa(model, eval_tf, list(paths))
            res["field"] = {
                "po_slici": [
                    {"fajl": str(ROOT.joinpath(path).relative_to(ROOT)).split("field_photos/")[-1],
                     "prava_klasa": ["zdrava", "varoa"][t], "p_varoa": round(float(pv), 4)}
                    for path, t, pv in zip(paths, y, p)
                ],
                **_metrics(np.array(y), p),
            }

        if bee:
            paths, y = zip(*bee)
            p = _p_varoa(model, eval_tf, list(paths))
            res["beedataset"] = _metrics(np.array(y), p)

        results[run] = res
        print(f"\n=== {config['arch_display']} ===")
        if "field" in res:
            for r in res["field"]["po_slici"]:
                guess = "varoa" if r["p_varoa"] >= 0.5 else "zdrava"
                ok = "✓" if guess == r["prava_klasa"] else "✗"
                print(f"  {ok} {r['fajl']:45s} prava: {r['prava_klasa']:6s} P(varoa) = {r['p_varoa']:.3f}")
        if "beedataset" in res:
            print("  BeeDataset:", res["beedataset"])

    out = RUNS / "field_test.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSnimljeno: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
