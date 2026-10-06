"""
VarroaDataset (Schurischuster & Kampel, TU Wien, 2020) — Zenodo 10.5281/zenodo.4085044.

Za razliku od BeeImage-a, oznaka je dodeljena SVAKOJ PČELI posebno, i to
vizuelno: pčela je označena kao zaražena ako se na njoj vidi grinja, a svaka
grinja ima ručno nacrtan pravougaonik (bounding box).

Format gt.csv (razdvojeno razmakom, bez zaglavlja):
    <split>/videos/<video>/<fajl>.png  <oznaka>  [x1 y1 x2 y2]*

  oznaka 0 — zdrava pčela
  oznaka 1 — vidljiva grinja
  oznaka 3 — nije dokumentovana; zvanična statistika autora (3.947 zaraženih)
             je zbir oznaka 1 i 3, a sve slike sa 3 imaju box grinje — tretira
             se kao zaražena. Najčešće je grinja delimično skrivena.

Podela train/val/test je zvanična, od autora, i ide PO VIDEU: nijedan video se
ne pojavljuje u dva skupa. Zato test meri rad na snimcima koje model nije video.

Koordinate grinja se NE koriste u treningu (klasifikacija koristi samo oznaku),
nego samo u proveri da li model zaista gleda grinju (src/model_checks.py).

    python -m src.varroa_dataset            # preuzimanje + statistika
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile

import pandas as pd

from .paths import VARROA_DIR, VARROA_GT, VARROA_ZENODO_RECORD

CLASSES = ["zdrava", "varoa"]
SPLITS = ["train", "val", "test"]
RAW_TO_TARGET = {0: 0, 1: 1, 3: 1}


def parse_gt() -> pd.DataFrame:
    if not VARROA_GT.is_file():
        raise FileNotFoundError(f"Nema {VARROA_GT}. Pokreni prvo: python -m src.varroa_dataset")

    rows = []
    for line in VARROA_GT.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        rel_path, raw = parts[0], int(parts[1])
        coords = [int(v) for v in parts[2:]]
        boxes = [tuple(coords[i:i + 4]) for i in range(0, len(coords), 4)]
        split, _, video = rel_path.split("/")[:3]
        rows.append({
            "file": rel_path,
            "split": split,
            "video": video,
            "raw_label": raw,
            "boxes": boxes,
            "n_mites": len(boxes),
        })

    df = pd.DataFrame(rows)
    unknown = set(df["raw_label"]) - set(RAW_TO_TARGET)
    if unknown:
        raise ValueError(f"Nepoznate oznake u gt.csv: {sorted(unknown)}")

    df["target"] = df["raw_label"].map(RAW_TO_TARGET)
    df["label"] = df["target"].map(lambda t: CLASSES[t])
    df["path"] = df["file"].map(lambda f: VARROA_DIR / f)
    return df


def load_varroa_split(split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"Nepoznat split '{split}', dostupni: {SPLITS}")
    df = parse_gt()
    df = df[df["split"] == split].reset_index(drop=True)

    missing = ~df["path"].map(lambda p: p.is_file())
    if missing.any():
        raise FileNotFoundError(
            f"{int(missing.sum())} slika iz '{split}' nema na disku "
            f"(npr. {df.loc[missing, 'path'].iloc[0]}). Pokreni: python -m src.varroa_dataset"
        )
    return df


def download() -> None:
    """Preuzima gt.csv i tri zip arhive sa Zenodo-a (~1,2 GB) i raspakuje ih."""
    VARROA_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://zenodo.org/api/records/{VARROA_ZENODO_RECORD}/files"

    if not VARROA_GT.is_file():
        print("Preuzimam gt.csv ...")
        urllib.request.urlretrieve(f"{base}/gt.csv/content", VARROA_GT)

    for split in SPLITS:
        if (VARROA_DIR / split / "videos").is_dir():
            print(f"{split}: već raspakovan, preskačem")
            continue
        archive = VARROA_DIR / f"{split}.zip"
        print(f"Preuzimam {split}.zip ...")
        urllib.request.urlretrieve(f"{base}/{split}.zip/content", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(VARROA_DIR)
        archive.unlink()


def describe() -> pd.DataFrame:
    df = parse_gt()
    print(f"Ukupno slika: {len(df)}")
    print("\nPo splitu i klasi:")
    print(pd.crosstab(df["split"], df["label"], margins=True).loc[SPLITS + ["All"]])
    print("\nOriginalne oznake (0/1/3):")
    print(pd.crosstab(df["split"], df["raw_label"]).loc[SPLITS])
    videos = df.groupby("video").agg(split=("split", "first"), slika=("target", "size"),
                                     udeo_varoe=("target", "mean"))
    print(f"\nVidea: {len(videos)}; video u više splitova: "
          f"{int((df.groupby('video')['split'].nunique() > 1).sum())}")
    print(videos.sort_values("split").round(2).to_string())
    return df


if __name__ == "__main__":
    download()
    describe()
    sys.exit(0)
