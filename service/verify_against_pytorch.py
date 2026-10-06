"""
Provera da servis daje iste rezultate kao model iz treninga.

Za svaku sliku iz test skupa poredi:
  - PyTorch: torchvision weights.transforms() + best.pt   (ono što je evaluirano)
  - Servis:  ručno preprocesiranje (PIL+numpy) + ONNX      (ono što se servira)

Prijavljuje najveću razliku u verovatnoćama, broj različitih predikcija, i
accuracy servisa na test skupu — koja mora biti ista kao u izveštaju.

    ./.venv/bin/python -m service.verify_against_pytorch
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from PIL import Image

from src.dataset import load_split
from src.models import build_model
from src.paths import MODELS, RUNS

from .inference import BeeHealthModel


def main() -> int:
    served = BeeHealthModel(MODELS / "bee_health.onnx")
    run = served.meta["source_run"]

    dataset = served.meta.get("dataset", "beeimage")
    mode = served.meta["preprocessing"].get("mode", "center_crop")
    model, _, eval_tf = build_model(served.meta["arch"], served.meta["num_classes"], mode,
                                  served.meta["preprocessing"].get("size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    model.eval()

    df = load_split("test", served.meta["num_classes"], dataset)
    max_diff, disagreements, correct = 0.0, 0, 0

    for path, target in zip(df["path"], df["target"]):
        with open(path, "rb") as f:
            result = served.predict(f.read())
        p_service = np.array([result["all_probs"][c] for c in served.classes])

        with torch.no_grad():
            x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0)
            p_torch = torch.softmax(model(x), dim=1)[0].numpy()

        max_diff = max(max_diff, float(np.abs(p_service - p_torch).max()))
        disagreements += int(p_service.argmax() != p_torch.argmax())
        correct += int(served.classes.index(result["label"]) == int(target))

    accuracy = correct / len(df)
    print(f"Slika u test skupu:              {len(df)}")
    print(f"Najveća razlika u verovatnoći:   {max_diff:.5f}")
    print(f"Različitih predikcija:           {disagreements}")
    print(f"Accuracy servisa na testu:       {accuracy:.4f}   (izveštaj: {served.meta['test_metrics']['accuracy']})")

    # Verovatnoće servisa su zaokružene na 4 decimale, pa je prag 1e-3.
    ok = max_diff < 1e-3 and disagreements == 0
    (MODELS / "verification.json").write_text(json.dumps({
        "model": served.arch, "source_run": run, "test_slika": len(df),
        "najveca_razlika_verovatnoce": round(max_diff, 6), "razlicitih_predikcija": disagreements,
        "accuracy_servisa": round(accuracy, 4), "ok": ok,
    }, indent=2, ensure_ascii=False))
    print("OK — servis se poklapa sa treningom." if ok else "GREŠKA — preprocesiranje se razlikuje!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
