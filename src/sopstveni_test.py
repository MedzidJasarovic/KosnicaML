"""
Evaluacija na SOPSTVENOM test skupu — fotografije pčela koje model nikad nije video,
sa drugim kamerama, pozadinama i osvetljenjem (data/sopstveni_skup/).

Ovo je nezavisni (spoljni) test skup: ne koristi se ni za trening ni za izbor
podešavanja, samo za ocenu. Oznake su napravljene alatom src/annotate.py:
klasa po folderu, pravougaonik oko pčele i oko svake grinje (oznake.json, gt.csv).

Sva tri modela se ocenjuju na dva načina:
  CELA SLIKA        — fotografija kakva jeste (ono što korisnik učita u aplikaciju)
  ISEČENO OKO PČELE — isečak oko označene pčele (+10% sa svake strane), kao što su
                      isečene slike u VarroaDataset-u. Razlika između ova dva pokazuje
                      koliko smeta kad je pčela mala u kadru.

Za svaki način:
  1. METRIKE — accuracy, precision, recall, F1, ROC AUC, matrica konfuzije
  2. BRISANJE GRINJE — svaka grinja se prekrije bojom okoline; kontrola: prekrije se
     isto toliko mesta na DRUGOM delu iste pčele (prosek 5 nasumičnih položaja)
  3. GRAD-CAM — da li najjača tačka mape pada na grinju ("pointing game")

    python -m src.sopstveni_test

Rezultati: runs/sopstveni_test.json, runs/sopstveni_po_slici.csv, grafici/06*.png
"""
from __future__ import annotations

import copy
import json
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .annotate import CLASSES, OWN_DIR, list_images, load_annotations
from .evaluate import load_run
from .model_checks import _cover, _letterbox_geometry, _p_varoa, grad_cam
from .models import build_model, pick_device
from .paths import ROOT, RUNS

SEED = 42
CROP_MARGIN = 0.10      # isečak oko pčele: +10% širine/visine pčele sa svake strane
CONTROL_REPEATS = 5     # kontrolno prekrivanje: prosek preko 5 nasumičnih položaja
GRAFICI = ROOT / "grafici"
VARIANTS = ["cela slika", "isečeno oko pčele"]
BOJA_KLASE = {"zdrava": "#2ecc71", "varoa": "#e74c3c"}
BOJA_MODELA = {"MobileNetV3-Large": "#3498db", "ResNet50": "#9b59b6", "EfficientNetB0": "#f39c12"}


# --- podaci -------------------------------------------------------------------

def load_items() -> list[dict]:
    """Slike sa oznakama. Prekida ako neka slika nije označena — rezultat bi bio nepotpun."""
    ann = load_annotations()
    items, missing = [], []
    for name in list_images():
        a = ann.get(name, {})
        target = CLASSES.index(name.split("/")[0])
        if not a.get("pcela") or (target == 1 and not a.get("grinje")):
            missing.append(name)
            continue
        items.append({"name": name, "target": target, "bee": tuple(a["pcela"]),
                      "mites": [tuple(b) for b in a.get("grinje", [])]})
    if missing:
        raise SystemExit(f"{len(missing)} slika nije označeno (npr. {missing[0]}). "
                         "Pokreni: python -m src.annotate")
    return items


def open_image(name: str) -> Image.Image:
    # exif_transpose: telefon često snimi sliku zarotiranu + oznaku za rotaciju;
    # browser (alat za označavanje) je poštuje, pa i ovde mora, da se poklope koordinate
    return ImageOps.exif_transpose(Image.open(OWN_DIR / name)).convert("RGB")


def crop_box(bee, size) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bee
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    w, h = size
    return (max(0, round(x1 - mx)), max(0, round(y1 - my)), min(w, round(x2 + mx)), min(h, round(y2 + my)))


def shift(boxes, crop) -> list[tuple]:
    return [(b[0] - crop[0], b[1] - crop[1], b[2] - crop[0], b[3] - crop[1]) for b in boxes]


def control_boxes(item, rng) -> list[tuple]:
    """Isto toliko pravougaonika iste veličine, unutar pčele, bez preklapanja sa grinjama."""
    bx1, by1, bx2, by2 = item["bee"]
    out = []
    for m in item["mites"]:
        bw, bh = m[2] - m[0], m[3] - m[1]
        cand = m
        for _ in range(100):
            x = rng.randint(bx1, max(bx1, bx2 - bw))
            y = rng.randint(by1, max(by1, by2 - bh))
            cand = (x, y, x + bw, y + bh)
            if not any(not (cand[2] <= b[0] or cand[0] >= b[2] or cand[3] <= b[1] or cand[1] >= b[3])
                       for b in item["mites"] + out):
                break
        out.append(cand)
    return out


def variant(img: Image.Image, item, which: str) -> Image.Image:
    return img if which == VARIANTS[0] else img.crop(crop_box(item["bee"], img.size))


# --- metrike ------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "slika": int(len(y)),
        "sa_varoom": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision_varoa": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_varoa": round(float(recall_score(y, pred, zero_division=0)), 4),
        "recall_zdrava": round(float(recall_score(1 - y, 1 - pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def evaluate_run(run: str, items: list[dict], images: list[Image.Image]) -> tuple[dict, dict]:
    config = load_run(run)
    arch = config["arch"]
    device = pick_device()
    model, _, eval_tf = build_model(arch, 2, "letterbox", config.get("img_size"))
    model.load_state_dict(torch.load(RUNS / run / "best.pt", map_location="cpu"))
    model.eval()
    cam_model = copy.deepcopy(model)  # Grad-CAM na CPU — backward hook-ovi su tu pouzdani
    model.to(device)
    size = eval_tf.crop_size[0]
    y = np.array([it["target"] for it in items])
    infected = [i for i, it in enumerate(items) if it["target"] == 1]

    res, probs, gradcam_examples = {"arch_display": config["arch_display"]}, {}, []
    for which in VARIANTS:
        # 1. metrike
        p = _p_varoa(model, eval_tf, [variant(im, it, which) for im, it in zip(images, items)], device)
        probs[which] = p
        out = {"metrike": metrics(y, p)}

        # 2. brisanje grinje (prekrivanje na celoj slici, pa isecanje — isti box za sve tri verzije)
        rng = random.Random(SEED)
        covered = [variant(_cover(images[i], items[i]["mites"]), items[i], which) for i in infected]
        p_cov = _p_varoa(model, eval_tf, covered, device)
        p_ctl = np.mean([_p_varoa(model, eval_tf,
                                  [variant(_cover(images[i], control_boxes(items[i], rng)), items[i], which)
                                   for i in infected], device)
                         for _ in range(CONTROL_REPEATS)], axis=0)
        p_inf = p[infected]
        out["brisanje_grinje"] = {
            "zarazenih_slika": len(infected),
            "prosecna_p_varoa_original": round(float(p_inf.mean()), 4),
            "prosecna_p_varoa_grinja_prekrivena": round(float(p_cov.mean()), 4),
            "prosecna_p_varoa_drugo_mesto_prekriveno": round(float(p_ctl.mean()), 4),
            "udeo_varoa_original": round(float((p_inf >= 0.5).mean()), 4),
            "udeo_varoa_grinja_prekrivena": round(float((p_cov >= 0.5).mean()), 4),
            "udeo_varoa_drugo_mesto_prekriveno": round(float((p_ctl >= 0.5).mean()), 4),
            "po_slici": [{"fajl": items[i]["name"], "original": round(float(a), 4),
                          "grinja_prekrivena": round(float(b), 4), "drugo_mesto": round(float(c), 4)}
                         for i, a, b, c in zip(infected, p_inf, p_cov, p_ctl)],
        }

        # 3. Grad-CAM pointing game — na svim zaraženim slikama (malo ih je, nema uzorkovanja)
        hits = 0
        for i in infected:
            img, boxes = images[i], items[i]["mites"]
            if which != VARIANTS[0]:
                crop = crop_box(items[i]["bee"], img.size)
                img, boxes = img.crop(crop), shift(boxes, crop)
            cam = grad_cam(cam_model, arch, eval_tf(img)[None])
            scale, ox, oy = _letterbox_geometry(img.size, size)
            yy, xx = np.unravel_index(cam.argmax(), cam.shape)
            px, py = (xx - ox) / scale, (yy - oy) / scale
            margin = 0.03 * max(img.size)  # Grad-CAM je gruba mapa (7x7 pre interpolacije)
            hit = any(b[0] - margin <= px <= b[2] + margin and b[1] - margin <= py <= b[3] + margin
                      for b in boxes)
            hits += hit
            if which == VARIANTS[1]:
                gradcam_examples.append((items[i]["name"], img, cam, boxes, scale, ox, oy, hit, p[i]))
        out["gradcam_pointing"] = {"slika": len(infected), "pogodak_u_grinju": round(hits / len(infected), 4)}
        res[which] = out

    res["_gradcam_examples"] = gradcam_examples
    return res, probs


# --- grafici ------------------------------------------------------------------

def _save(fig, name: str) -> None:
    GRAFICI.mkdir(exist_ok=True)
    fig.savefig(GRAFICI / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(items, images) -> None:
    cols = 6
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.5 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, it, img in zip(axes.flat, items, images):
        # smanjenje PIL-om (sa zaglađivanjem) — matplotlib bi veliku sliku smanjio "zrnasto"
        k = 400 / max(img.size)
        im = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([v * k for v in it["bee"]], outline=(31, 111, 209), width=2)
        for b in it["mites"]:
            d.rectangle([v * k for v in b], outline=(230, 30, 30), width=2)
        ax.imshow(im)
        cls = CLASSES[it["target"]]
        ax.set_title(it["name"].split("/")[1].removesuffix(".jpg"), fontsize=9, color=BOJA_KLASE[cls])
    fig.suptitle("Sopstveni test skup — plavo: pčela, crveno: grinja", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06a_sopstveni_skup")


def plot_per_image(items, all_probs) -> None:
    names = [it["name"].split("/")[1].removesuffix(".jpg") for it in items]
    x = np.arange(len(items))
    n_healthy = sum(1 for it in items if it["target"] == 0)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, which in zip(axes, VARIANTS):
        ax.axvspan(-0.5, n_healthy - 0.5, color=BOJA_KLASE["zdrava"], alpha=0.08)
        ax.axvspan(n_healthy - 0.5, len(items) - 0.5, color=BOJA_KLASE["varoa"], alpha=0.08)
        for k, (model, probs) in enumerate(all_probs.items()):
            ax.scatter(x + (k - 1) * 0.22, probs[which], color=BOJA_MODELA[model], s=28, label=model, zorder=3)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("P(varoa)")
        ax.set_title(f"{which} — iznad isprekidane linije model kaže „varoa“", loc="left")
        ax.text(n_healthy / 2 - 0.5, 0.93, "prave zdrave", ha="center", color="#1e8449")
        ax.text(n_healthy + (len(items) - n_healthy) / 2 - 0.5, 0.93, "prave sa varoom", ha="center", color="#b03a2e")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=9, frameon=False)
    axes[1].set_xticks(x, names, rotation=90, fontsize=8)
    fig.tight_layout()
    _save(fig, "06b_sopstveni_po_slici")


def plot_comparison(results) -> None:
    rows = []
    for run, res in results.items():
        m = res["arch_display"]
        vd = json.loads((RUNS / run / "metrics.json").read_text())
        rows.append((m, "VarroaDataset test", vd["macro_f1"], vd["varroa_recall"]))
        field = RUNS / "field_test.json"
        if field.is_file():
            bd = json.loads(field.read_text()).get(run, {}).get("beedataset")
            if bd:
                rows.append((m, "BeeDataset", bd["macro_f1"], bd["recall_varoa"]))
        for which in VARIANTS:
            mt = res[which]["metrike"]
            rows.append((m, f"sopstveni — {which}", mt["macro_f1"], mt["recall_varoa"]))
    df = pd.DataFrame(rows, columns=["model", "skup", "F1 macro", "recall varoa"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, col in zip(axes, ["F1 macro", "recall varoa"]):
        t = df.pivot(index="model", columns="skup", values=col)
        t = t.loc[list(BOJA_MODELA), [c for c in ["VarroaDataset test", "BeeDataset",
                                                  "sopstveni — cela slika",
                                                  "sopstveni — isečeno oko pčele"] if c in t.columns]]
        t.plot(kind="bar", ax=ax, rot=0, width=0.8, color=["#34495e", "#95a5a6", "#e67e22", "#1abc9c"])
        for p in ax.patches:
            h = p.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom",
                            fontsize=7, xytext=(0, 2), textcoords="offset points")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.1)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    fig.suptitle("Poređenje skupova: isti modeli, različito snimanje", fontweight="bold")
    fig.tight_layout()
    fig.legend(handles, labels, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    _save(fig, "06c_poredjenje_skupova")


def plot_confusion(results) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(3.6 * len(results), 6.6))
    for c, res in enumerate(results.values()):
        for r, which in enumerate(VARIANTS):
            cm = np.array(res[which]["metrike"]["confusion_matrix"])
            ax = axes[r, c]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                        color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], CLASSES)
            ax.set_yticks([0, 1], CLASSES)
            ax.set_xlabel("predviđeno")
            ax.set_ylabel("stvarno")
            ax.set_title(f"{res['arch_display']}\n{which}", fontsize=10, color=BOJA_MODELA[res["arch_display"]])
    fig.tight_layout()
    _save(fig, "06d_sopstveni_matrice_konfuzije")


def plot_masking(results) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, which in zip(axes, VARIANTS):
        data = {res["arch_display"]: [res[which]["brisanje_grinje"][k] for k in
                                      ["prosecna_p_varoa_original", "prosecna_p_varoa_grinja_prekrivena",
                                       "prosecna_p_varoa_drugo_mesto_prekriveno"]]
                for res in results.values()}
        t = pd.DataFrame(data, index=["original", "grinja prekrivena", "drugo mesto prekriveno"]).T
        t.plot(kind="bar", ax=ax, rot=0, width=0.75, color=["#e74c3c", "#2ecc71", "#95a5a6"])
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
        ax.set_title(which, loc="left")
        ax.set_ylabel("prosečna P(varoa) na pčelama sa varoom")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Brisanje grinje na sopstvenim slikama — da li model gleda grinju?", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06e_sopstveni_brisanje_grinje")


def plot_gradcam(examples, arch_display: str) -> None:
    examples = examples[:7]
    fig, axes = plt.subplots(2, len(examples), figsize=(2.3 * len(examples), 5.4))
    for i, (name, img, cam, boxes, scale, ox, oy, hit, p) in enumerate(examples):
        w, h = img.size
        crop = cam[oy:oy + round(h * scale), ox:ox + round(w * scale)]
        cam_img = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255
        boxed = img.copy()
        d = ImageDraw.Draw(boxed)
        for b in boxes:
            d.rectangle(b, outline=(255, 0, 0), width=max(2, round(max(w, h) / 150)))
        axes[0, i].imshow(boxed)
        axes[0, i].set_title(f"{name.split('/')[1].removesuffix('.jpg')}\nP(varoa) = {p:.2f}", fontsize=8)
        axes[1, i].imshow(img)
        axes[1, i].imshow(cam_img, cmap="jet", alpha=0.45)
        axes[1, i].set_title("pogodak" if hit else "promašaj", fontsize=8, color="green" if hit else "red")
        for ax in axes[:, i]:
            ax.axis("off")
    fig.suptitle(f"{arch_display} — Grad-CAM za klasu „varoa“ (isečeno oko pčele)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06f_sopstveni_gradcam")


# --- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    torch.manual_seed(SEED)
    items = load_items()
    images = [open_image(it["name"]) for it in items]
    n_inf = sum(it["target"] for it in items)
    print(f"Sopstveni skup: {len(items)} slika ({len(items) - n_inf} zdravih, {n_inf} sa varoom, "
          f"{sum(len(it['mites']) for it in items)} grinja)")

    runs = sorted(p.parent.name for p in RUNS.glob("varroa_*/config.json"))
    results, all_probs = {}, {}
    for run in runs:
        res, probs = evaluate_run(run, items, images)
        results[run] = res
        all_probs[res["arch_display"]] = probs
        print(f"\n=== {res['arch_display']} ===")
        for which in VARIANTS:
            m, b, g = res[which]["metrike"], res[which]["brisanje_grinje"], res[which]["gradcam_pointing"]
            print(f"  {which:18s} acc {m['accuracy']:.3f} | F1 {m['macro_f1']:.3f} | recall varoa "
                  f"{m['recall_varoa']:.3f} | recall zdrava {m['recall_zdrava']:.3f} | AUC {m['roc_auc']:.3f}")
            print(f"  {'':18s} P(varoa) original {b['prosecna_p_varoa_original']:.2f} → grinja prekrivena "
                  f"{b['prosecna_p_varoa_grinja_prekrivena']:.2f} / drugo mesto "
                  f"{b['prosecna_p_varoa_drugo_mesto_prekriveno']:.2f} | Grad-CAM na grinji "
                  f"{g['pogodak_u_grinju']:.0%}")

    # grafici (Grad-CAM primeri za model u produkciji)
    plot_dataset(items, images)
    plot_per_image(items, {m: all_probs[m] for m in BOJA_MODELA if m in all_probs})
    ordered = {r: results[r] for m in BOJA_MODELA for r in results if results[r]["arch_display"] == m}
    plot_comparison(ordered)
    plot_confusion(ordered)
    plot_masking(ordered)
    prod = next((r for r in results.values() if r["arch_display"] == "ResNet50"), None)
    if prod:
        plot_gradcam(prod["_gradcam_examples"], prod["arch_display"])

    for res in results.values():
        res.pop("_gradcam_examples")
    (RUNS / "sopstveni_test.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    per_image = pd.DataFrame({"fajl": [it["name"] for it in items],
                              "prava_klasa": [CLASSES[it["target"]] for it in items],
                              "grinja": [len(it["mites"]) for it in items]})
    for m, probs in all_probs.items():
        for which in VARIANTS:
            per_image[f"{m} | {which}"] = np.round(probs[which], 4)
    per_image.to_csv(RUNS / "sopstveni_po_slici.csv", index=False)
    print(f"\nSnimljeno: runs/sopstveni_test.json, runs/sopstveni_po_slici.csv, grafici/06a–06f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
