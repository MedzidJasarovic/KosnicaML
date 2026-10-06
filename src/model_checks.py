"""
Provera da li model zaista gleda grinju — a ne pozadinu, snimak ili nešto treće.
Ovo je odgovor na problem otkriven kod BeeImage modela (NALAZI.md, C-X).

VarroaDataset ima ručno označen pravougaonik svake grinje, pa su mogući testovi
koji na BeeImage-u nisu bili mogući:

  1. BRISANJE GRINJE — na zaraženoj test pčeli svaka grinja se prekrije bojom
     okoline. Ako model gleda grinju, procena treba da pređe u "zdrava".
     Kontrola: prekrije se isto toliko mesta na DRUGOM delu pčele — tu procena
     ne sme bitno da se promeni. Razlika između ova dva je dokaz.
  2. SAMO POZADINA — modelu se daju samo uglovi slike (25% × 25%). Pozadina
     je ista za obe klase, pa udeo "varoa" mora biti sličan za uglove zdravih i
     zaraženih pčela.
  3. GRAD-CAM — mapa delova slike koji su najviše doprineli odluci "varoa".
     "Pointing game": da li najjača tačka mape pada u pravougaonik grinje.

Rezultati idu u runs/<run>/checks.json, a slike u runs/<run>/gradcam.png.

    python -m src.model_checks --run varroa_mobilenet_v3_large
    python -m src.model_checks --all
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from .dataset import load_split
from .evaluate import load_run
from .models import build_model, pick_device
from .paths import RUNS

SEED = 42
BATCH = 64


def _box_area_on_bee(box, size, rng, avoid) -> tuple[int, int, int, int]:
    """Kontrolni pravougaonik iste veličine, na drugom mestu — ne preklapa NIJEDNU grinju."""
    w, h = size
    bw, bh = box[2] - box[0], box[3] - box[1]
    for _ in range(50):
        # srednji deo slike — tu je pčela; ivice su pozadina
        x = rng.randint(int(w * 0.15), max(int(w * 0.15), int(w * 0.85) - bw))
        y = rng.randint(int(h * 0.1), max(int(h * 0.1), int(h * 0.9) - bh))
        cand = (x, y, x + bw, y + bh)
        overlap = any(not (cand[2] <= b[0] or cand[0] >= b[2] or cand[3] <= b[1] or cand[1] >= b[3])
                      for b in avoid)
        if not overlap:
            return cand
    return cand


def _cover(image: Image.Image, boxes) -> Image.Image:
    """
    Prekriva pravougaonike bojom neposredne okoline (medijana prstena oko boxa),
    pa blago zamuti ivicu. Bez oštrih ivica koje bi same bile "signal".
    """
    img = image.copy()
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    draw = ImageDraw.Draw(img)
    mask = Image.new("L", img.size, 0)
    mdraw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        pad = 6
        X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        ring = np.ones((Y2 - Y1, X2 - X1), bool)
        ring[(y1 - Y1):(y2 - Y1), (x1 - X1):(x2 - X1)] = False
        color = np.median(arr[Y1:Y2, X1:X2][ring], axis=0) if ring.any() else arr.reshape(-1, 3).mean(0)
        draw.rectangle((x1, y1, x2, y2), fill=tuple(int(c) for c in color))
        mdraw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), fill=255)
    blurred = img.filter(ImageFilter.GaussianBlur(3))
    return Image.composite(blurred, img, mask.filter(ImageFilter.GaussianBlur(2)))


def _corners(image: Image.Image) -> list[Image.Image]:
    w, h = image.size
    qw, qh = w // 4, h // 4
    return [image.crop(b) for b in
            [(0, 0, qw, qh), (w - qw, 0, w, qh), (0, h - qh, qw, h), (w - qw, h - qh, w, h)]]


@torch.no_grad()
def _p_varoa(model, eval_tf, images: list[Image.Image], device) -> np.ndarray:
    out = []
    for i in range(0, len(images), BATCH):
        batch = torch.stack([eval_tf(im) for im in images[i:i + BATCH]]).to(device)
        out.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _target_layer(model, arch: str):
    return model.layer4 if arch == "resnet50" else model.features[-1]


def grad_cam(model, arch: str, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM za klasu 'varoa' (indeks 1). x: (1,3,H,W). Vraća mapu HxW u [0,1]."""
    acts, grads = {}, {}
    layer = _target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, 1].backward()
    finally:
        h1.remove()
        h2.remove()
    a, g = acts["a"][0], grads["g"][0]
    weights = g.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def _letterbox_geometry(size_wh, target: int):
    """Kako se original mapira u letterbox kvadrat: (scale, offset_x, offset_y)."""
    w, h = size_wh
    scale = target / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return scale, (target - nw) // 2, (target - nh) // 2


def check_run(run: str, gradcam_samples: int = 200) -> dict:
    config = load_run(run)
    if config.get("dataset") != "varroa":
        raise SystemExit(f"{run} nije VarroaDataset run — ove provere zahtevaju pravougaonike grinja.")

    arch = config["arch"]
    device = pick_device()  # predikcije (bez gradijenata) na MPS/GPU
    model, _, eval_tf = build_model(arch, 2, "letterbox", config.get("img_size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    model.eval()
    cam_model = copy.deepcopy(model)  # Grad-CAM na CPU — backward hook-ovi su tu pouzdani
    model.to(device)
    size = eval_tf.crop_size[0]

    df = load_split("test", dataset="varroa")
    rng = random.Random(SEED)
    infected = df[df["n_mites"] > 0].reset_index(drop=True)
    healthy = df[df["target"] == 0].reset_index(drop=True)
    print(f"\n=== {config['arch_display']} — provere na test skupu "
          f"({len(infected)} zaraženih, {len(healthy)} zdravih) ===")

    # --- 1. Brisanje grinje ---------------------------------------------------
    originals, covered, control = [], [], []
    for _, row in infected.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        originals.append(img)
        covered.append(_cover(img, row["boxes"]))
        control.append(_cover(img, [_box_area_on_bee(b, img.size, rng, row["boxes"]) for b in row["boxes"]]))

    p_orig = _p_varoa(model, eval_tf, originals, device)
    p_cov = _p_varoa(model, eval_tf, covered, device)
    p_ctl = _p_varoa(model, eval_tf, control, device)
    detected = p_orig >= 0.5
    masking = {
        "zarazenih_test_slika": int(len(infected)),
        "prepoznato_kao_varoa_original": round(float(detected.mean()), 4),
        "ostalo_varoa_kad_se_grinja_prekrije": round(float((p_cov[detected] >= 0.5).mean()), 4),
        "ostalo_varoa_kad_se_prekrije_drugo_mesto": round(float((p_ctl[detected] >= 0.5).mean()), 4),
        "prosecna_p_varoa_original": round(float(p_orig[detected].mean()), 4),
        "prosecna_p_varoa_grinja_prekrivena": round(float(p_cov[detected].mean()), 4),
        "prosecna_p_varoa_drugo_mesto_prekriveno": round(float(p_ctl[detected].mean()), 4),
    }
    print("1) brisanje grinje:", json.dumps(masking, ensure_ascii=False, indent=2))

    # --- 2. Samo pozadina (uglovi) -------------------------------------------
    def corner_rate(frame) -> float:
        crops = [c for p in frame["path"] for c in _corners(Image.open(p).convert("RGB"))]
        return float((_p_varoa(model, eval_tf, crops, device) >= 0.5).mean())

    background = {
        "udeo_varoa_uglovi_zdravih": round(corner_rate(healthy), 4),
        "udeo_varoa_uglovi_zarazenih": round(corner_rate(infected), 4),
    }
    print("2) samo pozadina:", background)

    # --- 3. Grad-CAM + pointing game -----------------------------------------
    sample = infected[detected].sample(min(gradcam_samples, int(detected.sum())), random_state=SEED)
    hits, examples = 0, []
    for _, row in sample.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        x = eval_tf(img)[None]
        cam = grad_cam(cam_model, arch, x)
        scale, ox, oy = _letterbox_geometry(img.size, size)
        yy, xx = np.unravel_index(cam.argmax(), cam.shape)
        px, py = (xx - ox) / scale, (yy - oy) / scale
        margin = 8  # px u originalu — Grad-CAM je gruba mapa (7x7 pre interpolacije)
        hit = any(b[0] - margin <= px <= b[2] + margin and b[1] - margin <= py <= b[3] + margin
                  for b in row["boxes"])
        hits += hit
        if len(examples) < 8:
            examples.append((img, cam, row["boxes"], scale, ox, oy, hit))
    pointing = {"uzoraka": int(len(sample)), "pogodak_u_grinju": round(hits / max(1, len(sample)), 4)}
    print("3) Grad-CAM pointing game:", pointing)

    _plot_gradcam(examples, size, f"{config['arch_display']} — Grad-CAM za klasu 'varoa'",
                  RUNS / run / "gradcam.png")
    _plot_masking_examples(model, eval_tf, infected, detected, device,
                           RUNS / run / "masking_examples.png", config["arch_display"])

    result = {"run": run, "arch_display": config["arch_display"],
              "masking": masking, "background": background, "gradcam_pointing": pointing}
    (RUNS / run / "checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _plot_gradcam(examples, size, title, out_path) -> None:
    fig, axes = plt.subplots(2, len(examples), figsize=(2.0 * len(examples), 6.4))
    for i, (img, cam, boxes, scale, ox, oy, hit) in enumerate(examples):
        # isečak cam-a koji odgovara originalu (bez dopune), vraćen na veličinu originala
        w, h = img.size
        crop = cam[oy:oy + round(h * scale), ox:ox + round(w * scale)]
        cam_img = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255
        boxed = img.copy()
        d = ImageDraw.Draw(boxed)
        for b in boxes:
            d.rectangle(b, outline=(255, 0, 0), width=2)
        axes[0, i].imshow(boxed)
        axes[1, i].imshow(img)
        axes[1, i].imshow(cam_img, cmap="jet", alpha=0.45)
        axes[1, i].set_title("pogodak" if hit else "promašaj", fontsize=8, color="green" if hit else "red")
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].set_title("grinja (box iz gt.csv)", fontsize=8, loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_masking_examples(model, eval_tf, infected, detected, device, out_path, arch_display) -> None:
    rows = infected[detected].sample(min(5, int(detected.sum())), random_state=SEED + 1)
    rng = random.Random(SEED)
    fig, axes = plt.subplots(3, len(rows), figsize=(2.0 * len(rows), 9))
    labels = ["original", "grinja prekrivena", "drugo mesto prekriveno"]
    for i, (_, row) in enumerate(rows.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        variants = [img, _cover(img, row["boxes"]),
                    _cover(img, [_box_area_on_bee(b, img.size, rng, row["boxes"]) for b in row["boxes"]])]
        probs = _p_varoa(model, eval_tf, variants, device)
        for r, (variant, p) in enumerate(zip(variants, probs)):
            axes[r, i].imshow(variant)
            axes[r, i].set_title(f"{labels[r]}\nP(varoa) = {p:.2f}", fontsize=8,
                                 color="red" if p >= 0.5 else "green")
            axes[r, i].axis("off")
    fig.suptitle(f"{arch_display} — šta se desi kad se grinja prekrije")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Provera da li model gleda grinju")
    parser.add_argument("--run")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    torch.manual_seed(SEED)

    runs = sorted(p.parent.name for p in RUNS.glob("varroa_*/config.json")) if args.all else [args.run]
    if not runs or runs == [None]:
        parser.error("navedi --run <ime> ili --all")
    for run in runs:
        check_run(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
