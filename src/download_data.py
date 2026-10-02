"""
Preuzima BeeImage dataset (Annotated Honey Bee Images) sa Kaggle-a u data/raw/.

Zahteva Kaggle API token:
    Kaggle -> Settings -> API -> Create New Token
    -> snimiti kao ~/.kaggle/kaggle.json, pa `chmod 600 ~/.kaggle/kaggle.json`

Ako token ne postoji, skripta ispisuje uputstvo za rucno preuzimanje
i prihvata vec raspakovan dataset na istom mestu.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from .paths import BEE_CSV, BEE_IMAGES_DIR, DATA_RAW, KAGGLE_DATASET

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"

MANUAL_HINT = f"""
Kaggle token nije nadjen na: {KAGGLE_JSON}

Opcija 1 (preporuceno) — API token:
  1. Registruj se / uloguj na https://www.kaggle.com
  2. Settings -> API -> "Create New Token"  (skida se kaggle.json)
  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
  4. Ponovo pokreni: python -m src.download_data

Opcija 2 — rucno preuzimanje:
  1. Otvori https://www.kaggle.com/datasets/{KAGGLE_DATASET}
  2. Klikni "Download" (ZIP)
  3. Raspakuj tako da postoje:
       {BEE_IMAGES_DIR}
       {BEE_CSV}
"""


def already_present() -> bool:
    return BEE_CSV.is_file() and BEE_IMAGES_DIR.is_dir() and any(BEE_IMAGES_DIR.iterdir())


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if already_present():
        n = sum(1 for _ in BEE_IMAGES_DIR.iterdir())
        print(f"Dataset je vec na mestu: {n} fajlova u {BEE_IMAGES_DIR}")
        return 0

    if not KAGGLE_JSON.is_file():
        print(MANUAL_HINT)
        return 1

    print(f"Preuzimam {KAGGLE_DATASET} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(DATA_RAW)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Preuzimanje nije uspelo: {exc}")
        print(MANUAL_HINT)
        return 1

    archives = list(DATA_RAW.glob("*.zip"))
    if not archives:
        print("ZIP arhiva nije nadjena posle preuzimanja.")
        return 1

    for archive in archives:
        print(f"Raspakujem {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_RAW)
        archive.unlink()

    if not already_present():
        print(f"Raspakovano, ali ocekivani fajlovi nisu nadjeni.\n{MANUAL_HINT}")
        return 1

    n = sum(1 for _ in BEE_IMAGES_DIR.iterdir())
    print(f"Gotovo: {n} slika u {BEE_IMAGES_DIR}, CSV u {BEE_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
