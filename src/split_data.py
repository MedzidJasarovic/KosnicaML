"""
Pravi podelu podataka na train/val/test i upisuje je u data/splits/.
Rezultat se komituje u git — svih 6 eksperimenata tako vidi identicne podatke.

Dve podele:

1. `train/val/test.csv` — STRATIFIKOVANA 70/15/15 po originalnih 6 klasa.
   Stratifikacija ide po 6 klasa (najfinija granularnost) da bi i eksperiment
   sa 3 klase koristio bas iste slike — inace poredjenje 6 vs 3 ne bi bilo fer.

2. `loc_train/loc_val/loc_test.csv` — DISJUNKTNA PO LOKACIJI.
   Slike u BeeImage-u dolaze iz malog broja lokacija i datuma, pa su mnogi
   isecci susedni frejmovi istog snimka. Slucajna podela zato moze staviti
   skoro identicne slike i u train i u test -> naduvana tacnost. Ova podela
   garantuje da se ni jedna lokacija ne pojavi u dva splita, i sluzi kao
   posteniji (pesimisticniji) test generalizacije.
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import load_metadata
from .paths import DATA_SPLITS

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def stratified_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70/15/15, stratifikovano po 6-klasnoj oznaci."""
    strat = df["label"]
    train_df, rest_df = train_test_split(
        df,
        test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=strat,
        random_state=SEED,
    )
    # Od ostatka (30%) pola u val, pola u test -> 15/15.
    val_df, test_df = train_test_split(
        rest_df,
        test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        stratify=rest_df["label"],
        random_state=SEED,
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def location_disjoint_split(df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """
    Cele lokacije se dodeljuju splitovima. Lokacije se sortiraju po velicini
    (najveca prva) i redom dodeljuju splitu koji je trenutno najdalje od svoje
    ciljane velicine — greedy raspodela koja priblizno pogadja 70/15/15.
    """
    if "location" not in df.columns:
        print("Upozorenje: nema kolone `location` — preskacem podelu po lokaciji.")
        return None

    sizes = df.groupby("location").size().sort_values(ascending=False)
    if len(sizes) < 3:
        print(f"Upozorenje: samo {len(sizes)} lokacija — podela po lokaciji nije moguca.")
        return None

    total = int(sizes.sum())
    targets = {
        "train": total * (1 - VAL_FRACTION - TEST_FRACTION),
        "val": total * VAL_FRACTION,
        "test": total * TEST_FRACTION,
    }
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}

    for location, count in sizes.items():
        # Split sa najvecim relativnim deficitom dobija ovu lokaciju.
        pick = max(current, key=lambda s: (targets[s] - current[s]) / targets[s])
        assigned[pick].append(location)
        current[pick] += int(count)

    out = {
        name: df[df["location"].isin(locs)].reset_index(drop=True)
        for name, locs in assigned.items()
    }
    for name, locs in assigned.items():
        print(f"  loc_{name}: {len(out[name])} slika iz {len(locs)} lokacija -> {locs}")
    return out


def report(name: str, splits: dict[str, pd.DataFrame]) -> None:
    print(f"\n--- {name} ---")
    table = pd.DataFrame(
        {split: part["label"].value_counts() for split, part in splits.items()}
    ).fillna(0).astype(int)
    table["ukupno"] = table.sum(axis=1)
    print(table.to_string())

    files = {split: set(part["file"]) for split, part in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = files[a] & files[b]
        status = "OK" if not overlap else f"GRESKA: {len(overlap)} preklapanja"
        print(f"  preklapanje {a}/{b}: {status}")

    missing = [s for s, part in splits.items() if part["label"].nunique() < table.shape[0]]
    if missing:
        print(f"  Upozorenje: splitovi {missing} ne sadrze sve klase.")


def main() -> int:
    df = load_metadata(num_classes=6)
    print(f"Ukupno slika sa oznakom i fajlom na disku: {len(df)}")
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)

    cols = ["file", "label"]

    splits = stratified_split(df)
    for name, part in splits.items():
        part[cols].to_csv(DATA_SPLITS / f"{name}.csv", index=False)
    report("Stratifikovana podela (glavna)", splits)

    loc_splits = location_disjoint_split(df)
    if loc_splits is not None:
        for name, part in loc_splits.items():
            part[cols].to_csv(DATA_SPLITS / f"loc_{name}.csv", index=False)
        report("Podela disjunktna po lokaciji (sekundarna)", loc_splits)

    print(f"\nSnimljeno u {DATA_SPLITS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
