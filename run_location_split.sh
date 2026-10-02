#!/usr/bin/env bash
# Sekundarni eksperiment: ista tri modela, ali na podeli DISJUNKTNOJ PO LOKACIJI.
# Pokazuje koliko tacnost pada kad u test setu nema slika iz lokacija koje je
# model video u treningu — posteniji test generalizacije na nov pcelinjak.
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python

for arch in mobilenet_v3_large resnet50 efficientnet_b0; do
  for k in 6 3; do
    echo; echo "--- $arch, $k klasa, podela po lokaciji ---"
    $PY -m src.train --arch "$arch" --classes "$k" --split-prefix loc_
  done
done

$PY -m src.evaluate --all
