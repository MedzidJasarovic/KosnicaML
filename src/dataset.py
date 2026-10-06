"""
BeeImage dataset: mapiranje zdravstvenih klasa i PyTorch Dataset.

Originalna `health` kolona u bee_data.csv ima 6 vrednosti. Pored eksperimenta
sa svih 6, radimo i grupisanje u 3 klase — jer klase kao "missing queen" imaju
samo dvocifren broj primera, pa su metrike na njima nepouzdane.
"""
from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .paths import BEE_CSV, BEE_IMAGES_DIR, DATA_SPLITS

# --- Originalnih 6 klasa -----------------------------------------------------
# Kljucevi su normalizovani (lowercase, bez viska praznina) jer u datasetu
# postoji tipfeler ("varrao") i nekonzistentna velicina slova.
HEALTH_6 = {
    "healthy": "healthy",
    "few varrao, hive beetles": "few_varroa_hive_beetles",
    "varroa, small hive beetles": "varroa_small_hive_beetles",
    "ant problems": "ant_problems",
    "hive being robbed": "hive_being_robbed",
    "missing queen": "missing_queen",
}

# --- Grupisanje u 3 klase ----------------------------------------------------
HEALTH_3 = {
    "healthy": "zdrava",
    "few varrao, hive beetles": "varoa_grinje",
    "varroa, small hive beetles": "varoa_grinje",
    "ant problems": "drugi_problem",
    "hive being robbed": "drugi_problem",
    "missing queen": "drugi_problem",
}

MAPPINGS = {6: HEALTH_6, 3: HEALTH_3}

# Fiksiran red klasa — MORA biti isti u treningu, evaluaciji i ONNX inferenciji.
CLASS_ORDER = {
    6: [
        "healthy",
        "few_varroa_hive_beetles",
        "varroa_small_hive_beetles",
        "ant_problems",
        "hive_being_robbed",
        "missing_queen",
    ],
    3: ["zdrava", "varoa_grinje", "drugi_problem"],
}

# Klase koje znace prisustvo varoe — recall na njima je najvaznija metrika,
# jer propustena varoa je najskuplja greska za pcelara.
VARROA_CLASSES = {
    6: ["few_varroa_hive_beetles", "varroa_small_hive_beetles"],
    3: ["varoa_grinje"],
}


def normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def load_metadata(num_classes: int = 6) -> pd.DataFrame:
    """Cita bee_data.csv i dodaje kolone `label` (string) i `target` (int)."""
    if num_classes not in MAPPINGS:
        raise ValueError(f"num_classes mora biti 3 ili 6, dobijeno: {num_classes}")
    if not BEE_CSV.is_file():
        raise FileNotFoundError(
            f"Nema {BEE_CSV}. Pokreni prvo: python -m src.download_data"
        )

    df = pd.read_csv(BEE_CSV)
    if "health" not in df.columns or "file" not in df.columns:
        raise ValueError(f"Neocekivane kolone u CSV-u: {list(df.columns)}")

    mapping = MAPPINGS[num_classes]
    df["health_norm"] = df["health"].map(normalize)

    unknown = sorted(set(df["health_norm"]) - set(mapping))
    if unknown:
        raise ValueError(
            "CSV sadrzi `health` vrednosti koje nisu u mapiranju: "
            f"{unknown}\nDopuni HEALTH_6/HEALTH_3 u src/dataset.py."
        )

    df["label"] = df["health_norm"].map(mapping)
    order = CLASS_ORDER[num_classes]
    df["target"] = df["label"].map({name: i for i, name in enumerate(order)})

    # Odbaci redove cija slika fizicki ne postoji na disku.
    df["path"] = df["file"].map(lambda f: BEE_IMAGES_DIR / str(f))
    missing = ~df["path"].map(lambda p: p.is_file())
    if missing.any():
        print(f"Upozorenje: {int(missing.sum())} redova iz CSV-a nema sliku na disku — preskacem ih.")
        df = df.loc[~missing].reset_index(drop=True)

    return df


DATASETS = ["beeimage", "varroa"]


def class_order(dataset: str, num_classes: int) -> list[str]:
    """Red klasa za dati dataset. VarroaDataset uvek ima 2 klase."""
    if dataset == "varroa":
        from .varroa_dataset import CLASSES
        return CLASSES
    return CLASS_ORDER[num_classes]


def varroa_classes(dataset: str, num_classes: int) -> list[str]:
    return ["varoa"] if dataset == "varroa" else VARROA_CLASSES[num_classes]


def load_split(split: str, num_classes: int = 6, dataset: str = "beeimage") -> pd.DataFrame:
    """Cita jedan od splitova (train/val/test) i mapira klase."""
    if dataset == "varroa":
        from .varroa_dataset import load_varroa_split
        return load_varroa_split(split)

    path = DATA_SPLITS / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Nema {path}. Pokreni prvo: python -m src.split_data"
        )
    split_df = pd.read_csv(path)
    full = load_metadata(num_classes)
    merged = full.merge(split_df[["file"]], on="file", how="inner")
    if len(merged) != len(split_df):
        print(
            f"Upozorenje: split '{split}' ima {len(split_df)} redova, "
            f"spojeno {len(merged)}."
        )
    return merged.reset_index(drop=True)


class BeeHealthDataset(Dataset):
    """Vraca (slika_tensor, target). Slike su varijabilnih dimenzija — transform resize-uje."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.targets = df["target"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]
