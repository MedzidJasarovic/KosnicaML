"""
Preuzima BeeImage dataset (Annotated Honey Bee Images) sa Kaggle-a u data/raw/.

Zahteva Kaggle autentifikaciju. Kaggle CLI 2.2+ prihvata (u ovom redu):
  - `kaggle auth login` (OAuth, bez tokena)
  - promenljivu okruzenja KAGGLE_API_TOKEN
  - fajl ~/.kaggle/access_token
  - stari ~/.kaggle/kaggle.json (username + key)

Ako nijedno nije prisutno, skripta ispisuje uputstvo za rucno preuzimanje
i prihvata vec raspakovan dataset na istom mestu.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from .paths import BEE_CSV, BEE_IMAGES_DIR, DATA_RAW, KAGGLE_DATASET

KAGGLE_DIR = Path.home() / ".kaggle"
# Kaggle CLI 2.2+ cita token iz access_token; kaggle.json je stari format.
CREDENTIAL_FILES = (KAGGLE_DIR / "access_token", KAGGLE_DIR / "kaggle.json")
OAUTH_CACHE = KAGGLE_DIR / "oauth_cache.json"

MANUAL_HINT = f"""
Kaggle autentifikacija nije nadjena.

Opcija 1 — API token u fajl:
  1. https://www.kaggle.com/settings/api -> "Generate New Token"
  2. printf '%s' '<token>' > {KAGGLE_DIR / "access_token"}
     chmod 600 {KAGGLE_DIR / "access_token"}
  3. Ponovo pokreni: python -m src.download_data

Opcija 2 — OAuth u browseru, bez tokena:
     kaggle auth login

Opcija 3 — rucno preuzimanje:
  1. Otvori https://www.kaggle.com/datasets/{KAGGLE_DATASET}
  2. Klikni "Download" (ZIP)
  3. Raspakuj tako da postoje:
       {BEE_IMAGES_DIR}
       {BEE_CSV}
"""


def has_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return True
    return any(p.is_file() for p in (*CREDENTIAL_FILES, OAUTH_CACHE))


def already_present() -> bool:
    return BEE_CSV.is_file() and BEE_IMAGES_DIR.is_dir() and any(BEE_IMAGES_DIR.iterdir())


def flatten_nested_images() -> None:
    """
    Kaggle ZIP raspakuje slike u bee_imgs/bee_imgs/ (dvostruko ugnezdeno).
    CSV referise samo imena fajlova, pa se unutrasnji folder podize nivo gore.
    """
    nested = BEE_IMAGES_DIR / BEE_IMAGES_DIR.name
    if not nested.is_dir():
        return
    print(f"Ravnam ugnezdenu strukturu {nested} -> {BEE_IMAGES_DIR}")
    staging = BEE_IMAGES_DIR.with_name(BEE_IMAGES_DIR.name + "_tmp")
    nested.rename(staging)
    BEE_IMAGES_DIR.rmdir()
    staging.rename(BEE_IMAGES_DIR)


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if already_present():
        flatten_nested_images()
        n = sum(1 for _ in BEE_IMAGES_DIR.iterdir())
        print(f"Dataset je vec na mestu: {n} fajlova u {BEE_IMAGES_DIR}")
        return 0

    if not has_credentials():
        print(MANUAL_HINT)
        return 1

    print(f"Preuzimam {KAGGLE_DATASET} ...")
    try:
        # sys.executable -m kaggle, a ne "kaggle" iz PATH-a: CLI je instaliran
        # u venv-u projekta i globalno ga moze i ne biti.
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", KAGGLE_DATASET, "-p", str(DATA_RAW)],
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

    flatten_nested_images()

    if not already_present():
        print(f"Raspakovano, ali ocekivani fajlovi nisu nadjeni.\n{MANUAL_HINT}")
        return 1

    n = sum(1 for _ in BEE_IMAGES_DIR.iterdir())
    print(f"Gotovo: {n} slika u {BEE_IMAGES_DIR}, CSV u {BEE_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
