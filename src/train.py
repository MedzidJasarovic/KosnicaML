"""
Trening jednog eksperimenta: jedna arhitektura x jedna sema klasa.

Sve osim backbone-a je identicno za sva tri modela (iste podele, batch,
augmentacija, optimizer, broj epoha, kriterijum izbora checkpointa) — to je
uslov da poredjenje bude fer, kako je profesorica trazila.

Trening ide u dve faze:
  faza 1 (feature extraction) — telo zamrznuto, trenira se samo nova glava
  faza 2 (fine-tuning)        — poslednji blok tela odmrznut, mali LR

Primer:
    python -m src.train --arch resnet50 --classes 3
    python -m src.train --arch resnet50 --classes 3 --split-prefix loc_
    python -m src.train --arch resnet50 --classes 3 --epochs-head 1 --epochs-finetune 0 --limit 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CLASS_ORDER, BeeHealthDataset, load_split
from .models import (
    ARCHITECTURES,
    build_model,
    count_parameters,
    display_name,
    freeze_backbone,
    head_parameters,
    pick_device,
    unfreeze_last_block,
)
from .paths import RUNS

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_name(arch: str, num_classes: int, split_prefix: str) -> str:
    suffix = "_loc" if split_prefix else ""
    return f"{arch}_{num_classes}cls{suffix}"


def class_weights(targets: list[int], num_classes: int, device) -> torch.Tensor:
    """
    Tezine obrnuto proporcionalne broju primera: total / (num_classes * count).
    Bez njih model naprosto nauci da uvek kaze "healthy" — ta klasa dominira.
    """
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # klasa koja ne postoji u train splitu
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(df, transform, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        BeeHealthDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    """Vraca (loss, accuracy, macro_f1) na datom loaderu."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            seen += targets.size(0)
            all_true.extend(targets.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / seen, correct / seen, macro_f1


def train_epochs(
    model, arch, train_loader, val_loader, criterion, device,
    epochs: int, lr: float, stage: str, history: list[dict], state: dict, out_dir,
) -> None:
    if epochs <= 0:
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total, trainable = count_parameters(model)
    print(f"\n[{stage}] epoha: {epochs}, lr: {lr:g}, trenirajucih parametara: {trainable:,} / {total:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"{stage} {epoch}/{epochs}", leave=False)

        for images, targets in bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)
            bar.set_postfix(loss=f"{running_loss / seen:.3f}")

        scheduler.step()
        train_loss, train_acc = running_loss / seen, correct / seen
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history.append({
            "stage": stage,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
        })

        flag = ""
        # Najbolji checkpoint se bira po macro F1, NE po accuracy — zbog disbalansa.
        if val_f1 > state["best_f1"]:
            state["best_f1"] = val_f1
            state["best_stage"] = stage
            state["best_epoch"] = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
            flag = "  <- najbolji"

        print(
            f"  {stage} {epoch}/{epochs}  train_loss {train_loss:.3f} acc {train_acc:.3f}  |  "
            f"val_loss {val_loss:.3f} acc {val_acc:.3f} macroF1 {val_f1:.3f}{flag}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Transfer learning nad BeeImage datasetom")
    parser.add_argument("--arch", required=True, choices=ARCHITECTURES)
    parser.add_argument("--classes", type=int, default=6, choices=[3, 6])
    parser.add_argument("--split-prefix", default="", choices=["", "loc_"],
                        help="'loc_' koristi podelu disjunktnu po lokaciji")
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: koristi samo N slika")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    classes = CLASS_ORDER[args.classes]
    name = run_name(args.arch, args.classes, args.split_prefix)
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {display_name(args.arch)} | {args.classes} klasa | device: {device} ===")

    train_df = load_split(f"{args.split_prefix}train", args.classes)
    val_df = load_split(f"{args.split_prefix}val", args.classes)
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(n=min(args.limit // 3 or 1, len(val_df)), random_state=args.seed)
    print(f"train: {len(train_df)} slika, val: {len(val_df)} slika")

    model, train_tf, eval_tf = build_model(args.arch, args.classes)
    model.to(device)

    train_loader = make_loader(train_df, train_tf, args.batch_size, True, args.workers)
    val_loader = make_loader(val_df, eval_tf, args.batch_size, False, args.workers)

    weights = class_weights(train_df["target"].astype(int).tolist(), args.classes, device)
    print("tezine klasa: " + ", ".join(f"{c}={w:.2f}" for c, w in zip(classes, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights)

    history: list[dict] = []
    state = {"best_f1": -1.0, "best_stage": None, "best_epoch": None}
    started = time.time()

    # --- Faza 1: telo zamrznuto, trenira se samo nova glava ---
    freeze_backbone(model, args.arch)
    train_epochs(model, args.arch, train_loader, val_loader, criterion, device,
                 args.epochs_head, args.lr_head, "head", history, state, out_dir)

    # --- Faza 2: poslednji blok tela odmrznut, mali LR ---
    unfreeze_last_block(model, args.arch)
    train_epochs(model, args.arch, train_loader, val_loader, criterion, device,
                 args.epochs_finetune, args.lr_finetune, "finetune", history, state, out_dir)

    elapsed = time.time() - started
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    total_params, _ = count_parameters(model)
    config = {
        "run": name,
        "arch": args.arch,
        "arch_display": display_name(args.arch),
        "num_classes": args.classes,
        "classes": classes,
        "split_prefix": args.split_prefix or "stratified",
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "seed": args.seed,
        "device": str(device),
        "total_params": total_params,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "best_val_macro_f1": round(state["best_f1"], 4),
        "best_stage": state["best_stage"],
        "best_epoch": state["best_epoch"],
        "train_seconds": round(elapsed, 1),
        "preprocessing": {
            "resize_size": list(eval_tf.resize_size),
            "crop_size": list(eval_tf.crop_size),
            "mean": list(eval_tf.mean),
            "std": list(eval_tf.std),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print(
        f"\nGotovo za {elapsed / 60:.1f} min. Najbolji val macro F1: {state['best_f1']:.4f} "
        f"({state['best_stage']} epoha {state['best_epoch']}). Snimljeno u {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
