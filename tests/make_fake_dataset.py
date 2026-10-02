"""
Pravi mali SINTETICKI dataset iste strukture kao BeeImage, za smoke test
celog lanca (podela -> EDA -> trening -> evaluacija -> ONNX) bez pravog
dataseta. Slike su sum, pa su metrike besmislene — proverava se samo da
pipeline radi end-to-end.

    python tests/make_fake_dataset.py <izlazni_folder>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HEALTH_VALUES = [
    "healthy",
    "few varrao, hive beetles",
    "Varroa, Small Hive Beetles",
    "ant problems",
    "hive being robbed",
    "missing queen",
]
LOCATIONS = ["Alvin, TX, USA", "Athens, GA, USA", "Des Moines, IA, USA", "Saratoga, CA, USA"]
PER_CLASS = 30


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_data").resolve()
    images_dir = out / "bee_imgs"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    for class_idx, health in enumerate(HEALTH_VALUES):
        for i in range(PER_CLASS):
            name = f"fake_{class_idx}_{i:03d}.png"
            # varijabilne dimenzije, kao u pravom datasetu
            w, h = int(rng.integers(40, 120)), int(rng.integers(40, 120))
            pixels = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images_dir / name)
            rows.append({
                "file": name,
                "date": f"2018-08-{(i % 28) + 1:02d}",
                "time": "12:00",
                "location": LOCATIONS[(class_idx * PER_CLASS + i) % len(LOCATIONS)],
                "zip code": 77511,
                "subspecies": "Italian honey bee",
                "health": health,
                "pollen_carrying": False,
                "caste": "worker",
            })

    pd.DataFrame(rows).to_csv(out / "bee_data.csv", index=False)
    print(f"Sinteticki dataset: {len(rows)} slika u {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
