"""
Priprema slike za model — JEDNA implementacija koju koriste i trening (src/) i
servis (service/). Zavisi samo od PIL-a i numpy-ja, bez PyTorch-a, da bi
Docker slika servisa ostala mala.

Letterbox: cela slika se smanji tako da duža strana bude `size`, pa se dopuni
do kvadrata neutralnom bojom. Za razliku od center crop-a, ništa se ne odseca —
kod uspravnih slika pčela (160×280) center crop bi odbacio glavu i zadak, a
grinja je često baš tamo.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

# ImageNet srednja boja: posle normalizacije daje vrednosti blizu 0,
# pa dopuna ne unosi "signal" u mrežu.
PAD_COLOR = (124, 116, 104)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    # fotografije sa telefona su često snimljene zarotirane, uz EXIF oznaku za rotaciju
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def to_input_array(image: Image.Image, mean, std) -> np.ndarray:
    """PIL (već letterbox-ovana) -> float32 niz (1, 3, H, W), normalizovan po kanalu."""
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
