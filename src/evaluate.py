"""
Evaluacija istreniranog modela na TEST setu — pokrece se tacno jednom po modelu,
na kraju. Val set je sluzio za izbor checkpointa; test se ne sme koristiti za
nikakvu odluku, inace rezultat nije posten.

Racuna sve sto je profesorica trazila: accuracy, precision, recall, F1
(po klasi + macro + weighted) i confusion matrix. Dodatno meri broj parametara
i latenciju inferencije na CPU — to je inzenjerski argument za izbor modela
koji ide u produkciju.

    python -m src.evaluate --run resnet50_3cls
    python -m src.evaluate --all
    python -m src.evaluate --summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .dataset import CLASS_ORDER, VARROA_CLASSES, BeeHealthDataset, load_split
from .models import build_model, count_parameters, pick_device
from .paths import RUNS


def load_run(run: str) -> dict:
    config_path = RUNS / run / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Nema {config_path} — da li je trening za '{run}' zavrsen?")
    return json.loads(config_path.read_text())


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vraca (y_true, y_pred, verovatnoce)."""
    model.eval()
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            prob = torch.softmax(logits, dim=1)
            trues.extend(targets.tolist())
            preds.extend(prob.argmax(dim=1).cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return np.array(trues), np.array(preds), np.array(probs)


def measure_cpu_latency(model, crop: int, runs: int = 30) -> float:
    """Prosecno vreme inferencije po slici na CPU, u milisekundama (batch=1)."""
    model = model.to("cpu").eval()
    sample = torch.randn(1, 3, crop, crop)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(sample)
        started = time.perf_counter()
        for _ in range(runs):
            model(sample)
        elapsed = time.perf_counter() - started
    return elapsed / runs * 1000


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out_path) -> None:
    """Confusion matrix normalizovana po redovima (procenat stvarne klase)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 3, 1.3 * len(classes) + 2.5))
    annotations = np.array([
        [f"{normalized[i, j]:.0%}\n({cm[i, j]})" for j in range(len(classes))]
        for i in range(len(classes))
    ])
    sns.heatmap(
        normalized, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1,
        xticklabels=classes, yticklabels=classes, cbar_kws={"label": "udeo u stvarnoj klasi"},
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predvidjena klasa")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_history(history: pd.DataFrame, title: str, out_path) -> None:
    """Krive ucenja — vertikalna linija oznacava prelaz iz faze 1 u fine-tuning."""
    history = history.reset_index(drop=True)
    x = np.arange(1, len(history) + 1)
    boundary = None
    if "finetune" in set(history["stage"]):
        boundary = int((history["stage"] == "head").sum()) + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(x, history["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(x, history["val_loss"], label="val", marker="o", ms=3)
    axes[0].set_ylabel("loss")
    axes[1].plot(x, history["val_acc"], label="val accuracy", marker="o", ms=3)
    axes[1].plot(x, history["val_macro_f1"], label="val macro F1", marker="o", ms=3)
    axes[1].set_ylabel("metrika")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel("epoha (kumulativno)")
        if boundary:
            ax.axvline(boundary, color="gray", ls="--", lw=1)
            ax.text(boundary, ax.get_ylim()[1], " fine-tuning", va="top", fontsize=8, color="gray")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_run(run: str) -> dict:
    config = load_run(run)
    arch, num_classes = config["arch"], config["num_classes"]
    classes = CLASS_ORDER[num_classes]
    out_dir = RUNS / run
    device = pick_device()

    print(f"\n=== {config['arch_display']} | {num_classes} klasa | podela: {config['split_prefix']} ===")

    model, _, eval_tf = build_model(arch, num_classes)
    state = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    prefix = "loc_" if config["split_prefix"] == "loc_" else ""
    test_df = load_split(f"{prefix}test", num_classes)
    loader = DataLoader(
        BeeHealthDataset(test_df, eval_tf), batch_size=32, shuffle=False, num_workers=4
    )
    print(f"test: {len(test_df)} slika")

    y_true, y_pred, _ = predict(model, loader, device)

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=classes, output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=classes, zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plot_confusion(
        cm, classes,
        f"{config['arch_display']} — {num_classes} klasa ({config['split_prefix']})",
        out_dir / "confusion_matrix.png",
    )

    history_path = out_dir / "history.csv"
    if history_path.is_file():
        plot_history(
            pd.read_csv(history_path),
            f"{config['arch_display']} — {num_classes} klasa",
            out_dir / "learning_curves.png",
        )

    # Recall na klasama varoe — najvaznija pojedinacna metrika za pcelara.
    varroa = [c for c in VARROA_CLASSES[num_classes] if c in classes]
    varroa_idx = [classes.index(c) for c in varroa]
    varroa_support = int(cm[varroa_idx, :].sum())
    varroa_recall = (
        float(cm[np.ix_(varroa_idx, varroa_idx)].sum() / varroa_support)
        if varroa_support else None
    )

    crop = eval_tf.crop_size[0] if isinstance(eval_tf.crop_size, (list, tuple)) else eval_tf.crop_size
    total_params, _ = count_parameters(model)
    latency_ms = measure_cpu_latency(model, crop)

    metrics = {
        "run": run,
        "arch": arch,
        "arch_display": config["arch_display"],
        "num_classes": num_classes,
        "split": config["split_prefix"],
        "test_size": int(len(test_df)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "varroa_recall": round(varroa_recall, 4) if varroa_recall is not None else None,
        "varroa_support": varroa_support,
        "best_val_macro_f1": config["best_val_macro_f1"],
        "total_params": total_params,
        "cpu_latency_ms": round(latency_ms, 1),
        "train_seconds": config["train_seconds"],
        "classes_absent_from_test": [classes[i] for i in range(num_classes) if i not in present],
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in classes
        },
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(
        f"accuracy {metrics['accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f} | "
        f"recall(varoa) {metrics['varroa_recall']} | {total_params/1e6:.1f}M param | "
        f"{latency_ms:.1f} ms/slika (CPU)"
    )
    return metrics


def build_summary() -> pd.DataFrame:
    """Zbirna tabela svih evaluiranih eksperimenata -> runs/summary.csv."""
    rows = []
    for metrics_path in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        rows.append({
            "arhitektura": m["arch_display"],
            "klase": m["num_classes"],
            "podela": m["split"],
            "accuracy": m["accuracy"],
            "macro_F1": m["macro_f1"],
            "weighted_F1": m["weighted_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "recall_varoa": m["varroa_recall"],
            "parametara_M": round(m["total_params"] / 1e6, 1),
            "latencija_ms": m["cpu_latency_ms"],
            "trening_min": round(m["train_seconds"] / 60, 1),
            "run": m["run"],
        })
    if not rows:
        print("Nema ni jednog metrics.json — pokreni prvo evaluaciju.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["klase", "macro_F1"], ascending=[True, False])
    out = RUNS / "summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSnimljeno: {out}")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluacija na test setu")
    parser.add_argument("--run", help="ime foldera u runs/, npr. resnet50_3cls")
    parser.add_argument("--all", action="store_true", help="evaluiraj sve istrenirane runove")
    parser.add_argument("--summary", action="store_true", help="samo sastavi zbirnu tabelu")
    args = parser.parse_args(argv)

    if args.summary:
        build_summary()
        return 0

    if args.all:
        runs = sorted(p.parent.name for p in RUNS.glob("*/best.pt"))
        if not runs:
            print("Nema istreniranih modela u runs/.")
            return 1
        for run in runs:
            evaluate_run(run)
        print("\n=== ZBIRNA TABELA ===")
        build_summary()
        return 0

    if not args.run:
        parser.error("navedi --run <ime>, --all ili --summary")
    evaluate_run(args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
