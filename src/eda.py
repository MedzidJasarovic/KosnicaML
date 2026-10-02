"""
Analiza dataseta (EDA) — prva stvar koju je profesorica trazila:
koje klase postoje, koliko ima primera po klasi, da li je skup pogodan.

Ispisuje tabele u markdownu (idu direktno u izvestaj) i snima grafikone
u report/figures/.

    python -m src.eda
    python -m src.eda --out-dir /tmp/proba      # ne pise u report/
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .dataset import CLASS_ORDER, HEALTH_3, HEALTH_6, load_metadata
from pathlib import Path

from .paths import FIGURES, REPORT


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def class_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabele broja slika po klasi, za 6 i za 3 klase."""
    counts6 = df["label"].value_counts()
    table6 = pd.DataFrame({
        "originalna `health` vrednost": [
            next(k for k, v in HEALTH_6.items() if v == name) for name in CLASS_ORDER[6]
        ],
        "klasa (6)": CLASS_ORDER[6],
        "broj slika": [int(counts6.get(name, 0)) for name in CLASS_ORDER[6]],
    })
    table6["udeo"] = (table6["broj slika"] / table6["broj slika"].sum()).map("{:.1%}".format)

    grouped = df["health_norm"].map(HEALTH_3)
    counts3 = grouped.value_counts()
    table3 = pd.DataFrame({
        "klasa (3)": CLASS_ORDER[3],
        "originalne klase koje ulaze": [
            ", ".join(sorted(k for k, v in HEALTH_3.items() if v == name))
            for name in CLASS_ORDER[3]
        ],
        "broj slika": [int(counts3.get(name, 0)) for name in CLASS_ORDER[3]],
    })
    table3["udeo"] = (table3["broj slika"] / table3["broj slika"].sum()).map("{:.1%}".format)
    return table6, table3


def plot_distribution(table6: pd.DataFrame, table3: pd.DataFrame, figures_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(table6["klasa (6)"], table6["broj slika"], color="#4878a8")
    axes[0].set_title("6 originalnih klasa")
    axes[1].barh(table3["klasa (3)"], table3["broj slika"], color="#c87137")
    axes[1].set_title("3 grupisane klase")
    for ax, table, col in ((axes[0], table6, "klasa (6)"), (axes[1], table3, "klasa (3)")):
        ax.set_xlabel("broj slika")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for y, value in enumerate(table["broj slika"]):
            ax.text(value, y, f" {value}", va="center", fontsize=9)
    fig.suptitle("Raspodela klasa u BeeImage datasetu")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, figures_dir, per_class: int = 5) -> None:
    classes = CLASS_ORDER[6]
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2 * per_class, 2.1 * len(classes)))
    for row, name in enumerate(classes):
        subset = df[df["label"] == name].head(per_class)
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(subset):
                ax.imshow(Image.open(subset.iloc[col]["path"]).convert("RGB"))
            if col == 0:
                ax.set_title(f"{name} (n={int((df['label'] == name).sum())})",
                             loc="left", fontsize=9)
    fig.suptitle("Primeri slika po klasi")
    fig.tight_layout()
    fig.savefig(figures_dir / "samples_per_class.png", dpi=130)
    plt.close(fig)


def image_size_stats(df: pd.DataFrame, sample: int = 500) -> pd.DataFrame:
    subset = df.sample(n=min(sample, len(df)), random_state=42)
    sizes = [Image.open(p).size for p in subset["path"]]
    widths = pd.Series([w for w, _ in sizes])
    heights = pd.Series([h for _, h in sizes])
    return pd.DataFrame({
        "dimenzija": ["sirina", "visina"],
        "min": [widths.min(), heights.min()],
        "median": [int(widths.median()), int(heights.median())],
        "max": [widths.max(), heights.max()],
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analiza BeeImage dataseta")
    parser.add_argument("--out-dir", help="gde snimiti izvestaj i grafikone (podrazumevano report/)")
    args = parser.parse_args(argv)

    report_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else REPORT
    figures_dir = report_dir / "figures" if args.out_dir else FIGURES
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(num_classes=6)

    table6, table3 = class_distribution(df)
    sizes = image_size_stats(df)

    locations = df["location"].value_counts() if "location" in df.columns else pd.Series(dtype=int)
    dates = df["date"].nunique() if "date" in df.columns else 0

    lines = [
        "# Analiza BeeImage dataseta",
        "",
        f"- Ukupno slika sa oznakom i fajlom na disku: **{len(df)}**",
        f"- Razlicitih lokacija: **{len(locations)}**",
        f"- Razlicitih datuma snimanja: **{dates}**",
        "",
        "## Raspodela po 6 originalnih klasa",
        "",
        md_table(table6),
        "",
        f"Najbrojnija klasa je {table6.loc[table6['broj slika'].idxmax(), 'klasa (6)']} "
        f"({table6['broj slika'].max()} slika), najmanja "
        f"{table6.loc[table6['broj slika'].idxmin(), 'klasa (6)']} "
        f"({table6['broj slika'].min()} slika) — odnos "
        f"**{table6['broj slika'].max() / max(table6['broj slika'].min(), 1):.0f} : 1**.",
        "",
        "## Raspodela po 3 grupisane klase",
        "",
        md_table(table3),
        "",
        "## Dimenzije slika",
        "",
        md_table(sizes),
        "",
        "Slike su varijabilnih dimenzija (isecci pojedinacnih pcela), pa ih svaki",
        "model skalira na svoju ulaznu rezoluciju preko zvanicnog `weights.transforms()`.",
        "",
        "## Raspodela po lokaciji",
        "",
        md_table(pd.DataFrame({
            "lokacija": locations.index,
            "broj slika": locations.values,
            "udeo": [f"{v / len(df):.1%}" for v in locations.values],
        })) if len(locations) else "_Nema kolone `location`._",
        "",
        "Mali broj lokacija je razlog zasto pored stratifikovane podele pravimo i",
        "podelu disjunktnu po lokaciji — susedni isecci istog snimka inace zavrse",
        "i u train i u test setu, pa je tacnost naduvana.",
        "",
        "## Grafikoni",
        "",
        "![Raspodela klasa](figures/class_distribution.png)",
        "",
        "![Primeri po klasi](figures/samples_per_class.png)",
        "",
    ]

    out = report_dir / "01_analiza_dataseta.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    plot_distribution(table6, table3, figures_dir)
    plot_samples(df, figures_dir)

    print("\n".join(lines[:40]))
    print(f"\nSnimljeno: {out}")
    print(f"Grafikoni: {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
