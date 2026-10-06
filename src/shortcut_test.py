"""
Test "prečice" (shortcut learning): da li model prepoznaje pčelu ili snimak?

Za svaku test sliku modelu se daju samo njena četiri ugla (25% širine i visine).
Pčela je u centru isečka, pa uglovi sadrže uglavnom pozadinu snimka. Ako je
tačnost na uglovima mnogo veća od slučajnih 33%, model klasu određuje po
pozadini, a ne po izgledu pčele.

    ./.venv/bin/python -m src.shortcut_test
"""
from __future__ import annotations

import io

from PIL import Image

from service.inference import BeeHealthModel

from .dataset import load_split
from .paths import MODELS


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> None:
    model = BeeHealthModel(MODELS / "beeimage" / "mobilenet_v3_large_3cls.onnx")
    df = load_split("test", model.meta["num_classes"])

    print(f"{'prava klasa':15s} {'slika':>5s}  {'cela slika':>10s}  {'samo uglovi':>11s}")
    for cls in model.classes:
        sub = df[df.label == cls]
        full = corners = 0.0
        for path in sub.path:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            qw, qh = w // 4, h // 4
            full += model.predict(_png(img))["label"] == cls
            boxes = [(0, 0, qw, qh), (w - qw, 0, w, qh), (0, h - qh, qw, h), (w - qw, h - qh, w, h)]
            corners += sum(model.predict(_png(img.crop(b)))["label"] == cls for b in boxes) / 4
        print(f"{cls:15s} {len(sub):5d}  {full / len(sub):10.1%}  {corners / len(sub):11.1%}")
    print("\nSlučajno pogađanje sa 3 klase = 33%.")


if __name__ == "__main__":
    main()
