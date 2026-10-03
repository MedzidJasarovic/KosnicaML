"""
Meri latenciju inferencije u onnxruntime-u — runtime-u koji se ZAPRAVO izvrsava
u produkcijskom FastAPI servisu.

Zasto zaseban merenje: evaluate.py meri latenciju u PyTorch-u, a PyTorch i
onnxruntime imaju bitno razlicito optimizovane CPU kernele. Za arhitekture sa
depthwise-separable konvolucijama (MobileNetV3, EfficientNetB0) razlika je
visestruka, pa bi odluka o izboru modela na osnovu PyTorch merenja bila
pogresna. Mera koja se brani je mera u runtime-u koji ide u produkciju.

    python -m src.benchmark_onnx
    python -m src.benchmark_onnx --runs 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd

from .paths import MODELS, RUNS

WARMUP = 10


def bench(session, crop: int, batch: int, runs: int) -> np.ndarray:
    """Vraca vremena po SLICI u milisekundama."""
    x = np.random.randn(batch, 3, crop, crop).astype("float32")
    for _ in range(WARMUP):
        session.run(None, {"input": x})
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, {"input": x})
        times.append((time.perf_counter() - started) / batch)
    return np.array(times) * 1000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Latencija ONNX modela")
    parser.add_argument("--runs", type=int, default=60)
    args = parser.parse_args(argv)

    models = sorted(MODELS.glob("*.onnx"))
    if not models:
        print("Nema ni jednog .onnx modela u models/ — pokreni prvo src.export_onnx.")
        return 1

    rows = []
    for path in models:
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.is_file():
            print(f"Preskacem {path.name}: nema {meta_path.name}")
            continue
        meta = json.loads(meta_path.read_text())
        crop = meta["preprocessing"]["center_crop"]

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        single = bench(session, crop, 1, args.runs)
        batched = bench(session, crop, 8, max(args.runs // 4, 5))

        # latencija u PyTorch-u iz evaluate.py, za poredjenje
        torch_ms = None
        metrics_path = RUNS / meta["source_run"] / "metrics.json"
        if metrics_path.is_file():
            torch_ms = json.loads(metrics_path.read_text()).get("cpu_latency_ms")

        rows.append({
            "model": meta["arch_display"],
            "klase": meta["num_classes"],
            "onnx_batch1_ms": round(float(single.mean()), 2),
            "onnx_median_ms": round(float(np.median(single)), 2),
            "onnx_p95_ms": round(float(np.percentile(single, 95)), 2),
            "onnx_batch8_ms_po_slici": round(float(batched.mean()), 2),
            "pytorch_batch1_ms": torch_ms,
            "ubrzanje_onnx": round(torch_ms / float(single.mean()), 1) if torch_ms else None,
            "fajl_MB": round(path.stat().st_size / 1e6, 1),
            "run": meta["source_run"],
        })

    df = pd.DataFrame(rows).sort_values("onnx_batch1_ms")
    out = MODELS / "latency_benchmark.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSnimljeno: {out}")
    print(f"onnxruntime {ort.__version__}, CPUExecutionProvider, {args.runs} merenja + {WARMUP} warmup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
