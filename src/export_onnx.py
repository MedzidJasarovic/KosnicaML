"""
Export izabranog modela u ONNX + metapodaci, i verifikacija da ONNX daje
iste verovatnoce kao PyTorch.

Uz .onnx se OBAVEZNO snima i .meta.json sa redom klasa i tacnim parametrima
preprocesiranja. Bez toga je model bezvredan: inferencija mora resize-ovati i
normalizovati sliku identicno treningu, inace su predikcije smece.

    python -m src.export_onnx --run resnet50_3cls
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from .dataset import BeeHealthDataset, class_order, load_split
from .evaluate import load_run
from .models import build_model
from .paths import MODELS, RUNS
from .train import preprocessing_for
from service.preprocessing import PAD_COLOR

TOLERANCE = 1e-4
OPSET = 18


def export(run: str, name: str | None = None, verify_images: int = 20) -> int:
    config = load_run(run)
    arch, num_classes = config["arch"], config["num_classes"]
    dataset = config.get("dataset", "beeimage")
    classes = class_order(dataset, num_classes)
    mode = preprocessing_for(dataset)

    model, _, eval_tf = build_model(arch, num_classes, mode, config.get("img_size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    model.eval()

    crop = eval_tf.crop_size[0] if isinstance(eval_tf.crop_size, (list, tuple)) else eval_tf.crop_size
    resize = eval_tf.resize_size[0] if isinstance(eval_tf.resize_size, (list, tuple)) else eval_tf.resize_size

    MODELS.mkdir(parents=True, exist_ok=True)
    stem = name or "bee_health"
    onnx_path = MODELS / f"{stem}.onnx"
    meta_path = MODELS / f"{stem}.meta.json"

    dummy = torch.randn(1, 3, crop, crop)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        # Dinamicki batch, da servis moze da obradi i vise slika odjednom.
        # Tuple forma (ne dict) — kod dict-a kljucevi moraju da se zovu kao
        # argumenti metode forward(), a to se razlikuje po arhitekturi.
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        opset_version=OPSET,
        # KRITICNO: podrazumevano torch izdvaja tezine u zaseban .onnx.data fajl.
        # Tada kopiranje samo .onnx u Docker daje model bez tezina. Jedan
        # samostalan fajl je jedini bezbedan oblik za deployment.
        external_data=False,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Exportovano: {onnx_path} ({size_mb:.1f} MB, samostalan fajl)")

    stray = onnx_path.with_suffix(".onnx.data")
    if stray.is_file():
        stray.unlink()
        print(f"Obrisan zaostali {stray.name} iz prethodnog exporta.")

    meta = {
        "model_version": stem,
        "source_run": run,
        "arch": arch,
        "arch_display": config["arch_display"],
        "num_classes": num_classes,
        # Red klasa MORA biti isti kao u treningu — indeks izlaza mapira se ovde.
        "classes": classes,
        "input_name": "input",
        "output_name": "logits",
        "output_activation": "softmax",
        "dataset": dataset,
        "preprocessing": {
            "mode": mode,
            # letterbox: duža strana -> size, dopuna do kvadrata bojom pad_color
            # center_crop: kraća strana -> resize_shorter_side, pa isecanje center_crop
            **({"size": int(crop), "pad_color": list(PAD_COLOR)} if mode == "letterbox"
               else {"resize_shorter_side": int(resize), "center_crop": int(crop)}),
            "rescale": "pixel / 255.0",
            "mean": [float(v) for v in eval_tf.mean],
            "std": [float(v) for v in eval_tf.std],
            "channel_order": "RGB",
            "layout": "NCHW",
        },
        "test_metrics": _test_metrics(run),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Metapodaci: {meta_path}")

    return verify(onnx_path, model, eval_tf, num_classes, config, verify_images)


def _test_metrics(run: str) -> dict | None:
    path = RUNS / run / "metrics.json"
    if not path.is_file():
        return None
    m = json.loads(path.read_text())
    return {k: m[k] for k in ("accuracy", "macro_f1", "varroa_recall", "roc_auc") if k in m}


def verify(onnx_path, model, eval_tf, num_classes: int, config: dict, count: int) -> int:
    """Uporedjuje ONNX i PyTorch verovatnoce na stvarnim test slikama."""
    prefix = "loc_" if config["split_prefix"] == "loc_" else ""
    df = load_split(f"{prefix}test", num_classes, config.get("dataset", "beeimage")).head(count)
    loader = DataLoader(BeeHealthDataset(df, eval_tf), batch_size=count, shuffle=False)
    images, _ = next(iter(loader))

    with torch.no_grad():
        torch_probs = torch.softmax(model(images), dim=1).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": images.numpy()})[0]
    onnx_probs = np.exp(onnx_logits - onnx_logits.max(axis=1, keepdims=True))
    onnx_probs /= onnx_probs.sum(axis=1, keepdims=True)

    max_diff = float(np.abs(torch_probs - onnx_probs).max())
    agree = int((torch_probs.argmax(1) == onnx_probs.argmax(1)).sum())

    print(
        f"\nVerifikacija na {len(df)} test slika: "
        f"maksimalna razlika verovatnoca {max_diff:.2e}, "
        f"iste predikcije {agree}/{len(df)}"
    )
    if max_diff > TOLERANCE or agree != len(df):
        print(f"GRESKA: razlika je iznad tolerancije {TOLERANCE:.0e}.")
        return 1
    print("OK — ONNX i PyTorch se poklapaju.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export modela u ONNX")
    parser.add_argument("--run", required=True, help="npr. resnet50_3cls")
    parser.add_argument("--name", help="osnovno ime izlaznih fajlova (podrazumevano bee_health)")
    parser.add_argument("--verify-images", type=int, default=20)
    args = parser.parse_args(argv)
    return export(args.run, args.name, args.verify_images)


if __name__ == "__main__":
    sys.exit(main())
