"""
Generiše sekciju sa rezultatima iz artefakata treninga (runs/*/metrics.json)
i sastavlja kompletan izveštaj report/izvestaj.md.

Tabele se ISKLJUČIVO čitaju iz metrics.json — nijedan broj se ne upisuje ručno.

    python -m src.make_report
"""
from __future__ import annotations

import json
import sys
from datetime import date

import pandas as pd

from .dataset import CLASS_ORDER
from .paths import REPORT, RUNS

# Redosled sekcija u konačnom dokumentu. Sekcija rezultata se generiše.
SECTIONS = [
    "01_analiza_dataseta.md",
    "02_transfer_learning.md",
    "03_pytorch_vs_keras.md",
    "04_rezultati.md",
    "05_izbor_modela.md",
    "06_integracija.md",
    "07_ogranicenja.md",
]

ARCH_ORDER = ["MobileNetV3-Large", "ResNet50", "EfficientNetB0"]


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def load_all_metrics() -> list[dict]:
    metrics = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*/metrics.json"))]
    if not metrics:
        raise SystemExit("Nema runs/*/metrics.json — pokreni prvo: python -m src.evaluate --all")
    return metrics


def comparison_table(metrics: list[dict], num_classes: int) -> pd.DataFrame:
    rows = []
    for m in metrics:
        if m["num_classes"] != num_classes:
            continue
        rows.append({
            "Arhitektura": m["arch_display"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Weighted F1": f"{m['weighted_f1']:.4f}",
            "Macro precision": f"{m['macro_precision']:.4f}",
            "Macro recall": f"{m['macro_recall']:.4f}",
            "Recall (varoa)": f"{m['varroa_recall']:.4f}" if m["varroa_recall"] is not None else "—",
            "Parametara": f"{m['total_params'] / 1e6:.1f}M",
            "Latencija (ms)": f"{m['cpu_latency_ms']:.1f}",
            "Trening (min)": f"{m['train_seconds'] / 60:.1f}",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_o"] = df["Arhitektura"].map({a: i for i, a in enumerate(ARCH_ORDER)})
    return df.sort_values("_o").drop(columns="_o")


def per_class_table(m: dict) -> pd.DataFrame:
    rows = []
    for name in CLASS_ORDER[m["num_classes"]]:
        pc = m["per_class"][name]
        rows.append({
            "Klasa": name,
            "Precision": f"{pc['precision']:.3f}",
            "Recall": f"{pc['recall']:.3f}",
            "F1": f"{pc['f1']:.3f}",
            "Primera u testu": pc["support"],
        })
    return pd.DataFrame(rows)


def build_results_section(metrics: list[dict]) -> str:
    lines = [
        "# Eksperimentalni rezultati",
        "",
        "## Uslovi eksperimenta",
        "",
        "Sve arhitekture trenirane su pod identičnim uslovima; razlikuje se",
        "isključivo telo mreže.",
        "",
        md_table(pd.DataFrame({
            "Parametar": [
                "Podela podataka", "Ulazna rezolucija", "Veličina batch-a",
                "Faza 1 (glava)", "Faza 2 (fine-tuning)", "Optimizator",
                "Raspored stope učenja", "Funkcija greške", "Augmentacija",
                "Izbor checkpointa", "Seed", "Uređaj",
            ],
            "Vrednost": [
                "stratifikovano 70/15/15, isti fajlovi za sve modele",
                "prema `weights.transforms()` svake arhitekture",
                "32",
                "15 epoha, telo zamrznuto, lr 1e-3",
                "10 epoha, poslednji blok odmrznut, lr 1e-5",
                "AdamW, weight decay 1e-4",
                "CosineAnnealingLR",
                "CrossEntropyLoss sa težinama klasa",
                "RandomResizedCrop, flip, rotacija 15°, ColorJitter 0.2",
                "najveći macro F1 na validacionom skupu",
                "42",
                "Apple M5, PyTorch MPS",
            ],
        })),
        "",
    ]

    for k in (6, 3):
        table = comparison_table(metrics, k)
        if table.empty:
            continue
        subset = [m for m in metrics if m["num_classes"] == k]
        test_size = subset[0]["test_size"]
        lines += [
            f"## Poređenje arhitektura — {k} klasa",
            "",
            f"Test skup: {test_size} slika.",
            "",
            md_table(table),
            "",
        ]
        for m in sorted(subset, key=lambda x: ARCH_ORDER.index(x["arch_display"])):
            lines += [
                f"### {m['arch_display']} — metrike po klasi ({k} klasa)",
                "",
                md_table(per_class_table(m)),
                "",
                f"![Confusion matrix]({RUNS.name}/{m['run']}/confusion_matrix.png)",
                "",
                f"![Krive učenja]({RUNS.name}/{m['run']}/learning_curves.png)",
                "",
            ]
            absent = m.get("classes_absent_from_test")
            if absent:
                lines += [
                    f"Klase bez ijednog primera u test skupu: {', '.join(absent)}.",
                    "",
                ]

    return "\n".join(lines)


def assemble() -> None:
    parts = [
        "---",
        'title: "Prepoznavanje zdravstvenog stanja pčela na osnovu fotografija"',
        'subtitle: "Transfer learning i poređenje arhitektura — izveštaj o dosadašnjem radu"',
        'author: "Medžid Jašarović"',
        f'date: "{date.today().strftime("%d.%m.%Y.")}"',
        "lang: sr",
        "toc: true",
        "---",
        "",
    ]
    missing = []
    for name in SECTIONS:
        path = REPORT / name
        if not path.is_file():
            missing.append(name)
            continue
        parts.append(path.read_text(encoding="utf-8").rstrip())
        parts.append("")

    out = REPORT / "izvestaj.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Sastavljeno: {out}")
    if missing:
        print(f"Nedostaju sekcije (preskočene): {', '.join(missing)}")


def main() -> int:
    metrics = load_all_metrics()
    section = build_results_section(metrics)
    (REPORT / "04_rezultati.md").write_text(section, encoding="utf-8")
    print(f"Generisano: {REPORT / '04_rezultati.md'} ({len(metrics)} eksperimenata)")
    assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
