#!/usr/bin/env bash
# Pokrece ceo eksperimentalni deo nad PRAVIM datasetom:
# EDA -> podela -> 6 treninga -> evaluacija -> zbirna tabela.
#
# Pretpostavlja da je dataset preuzet (python -m src.download_data).
# Procena trajanja na Apple M5: oko 1-2 sata.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "### Analiza dataseta"
$PY -m src.eda

echo; echo "### Podela podataka"
$PY -m src.split_data

echo; echo "### Trening: 3 arhitekture x 2 seme klasa"
for arch in mobilenet_v3_large resnet50 efficientnet_b0; do
  for k in 6 3; do
    echo; echo "--- $arch, $k klasa ---"
    $PY -m src.train --arch "$arch" --classes "$k"
  done
done

echo; echo "### Evaluacija na test setu + zbirna tabela"
$PY -m src.evaluate --all

echo
echo "Gotovo. Zbirna tabela: runs/summary.csv"
echo "Sledeci korak: izabrati najbolji run i exportovati ga:"
echo "  $PY -m src.export_onnx --run <ime_runa>"
