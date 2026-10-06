"""
Učitavanje ONNX modela i predikcija nad jednom slikom.

Preprocesiranje MORA davati isti tensor kao u treningu. Način se čita iz
<model>.meta.json, koji nastaje pri izvozu modela:

  "letterbox"   (VarroaDataset) — service/preprocessing.py, ista funkcija
                koju koristi i trening: duža strana na `size`, dopuna do kvadrata
  "center_crop" (BeeImage)      — kao torchvision `weights.transforms()`:
                kraća strana na `resize_shorter_side`, pa centralni isečak

Posle toga: piksel / 255, (x - mean) / std, HWC -> CHW, batch -> (1, 3, 224, 224).
Da se ovo poklapa sa PyTorch-om proverava `service/verify_against_pytorch.py`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .preprocessing import letterbox, to_input_array


class BeeHealthModel:
    def __init__(self, onnx_path: Path):
        meta_path = onnx_path.with_suffix(".meta.json")
        self.meta = json.loads(meta_path.read_text())
        self.classes: list[str] = self.meta["classes"]
        self.version: str = self.meta["model_version"]
        self.arch: str = self.meta["arch_display"]

        pre = self.meta["preprocessing"]
        self.mode = pre.get("mode", "center_crop")
        if self.mode == "letterbox":
            self.size = int(pre["size"])
        else:
            self.resize = int(pre["resize_shorter_side"])
            self.crop = int(pre["center_crop"])
        self.mean = np.array(pre["mean"], dtype=np.float32)
        self.std = np.array(pre["std"], dtype=np.float32)

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.meta["input_name"]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        if self.mode == "letterbox":
            return to_input_array(letterbox(image, self.size), self.mean, self.std)

        image = image.convert("RGB")

        # 1. Resize kraće strane — ista formula za dužu stranu kao u torchvision-u
        w, h = image.size
        if w <= h:
            new_w, new_h = self.resize, int(self.resize * h / w)
        else:
            new_w, new_h = int(self.resize * w / h), self.resize
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # 2. Center crop — torchvision zaokružuje pomeraj sa round()
        top = int(round((new_h - self.crop) / 2.0))
        left = int(round((new_w - self.crop) / 2.0))
        image = image.crop((left, top, left + self.crop, top + self.crop))

        # 3–5. Skaliranje, normalizacija, CHW, batch
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes))
        logits = self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]

        # softmax, numerički stabilan
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        best = int(probs.argmax())
        return {
            "label": self.classes[best],
            "confidence": round(float(probs[best]), 4),
            "all_probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "model_version": self.version,
        }
