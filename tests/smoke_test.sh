#!/usr/bin/env bash
# Smoke test celog lanca nad SINTETICKIM datasetom (slike su sum, pa su metrike
# besmislene — proverava se samo da pipeline radi end-to-end).
# Ne dira pravi dataset ni prave rezultate: sve ide u privremeni folder.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=./.venv/bin/python
TMP="${TMPDIR:-/tmp}/kosnica_ml_smoke"
rm -rf "$TMP"; mkdir -p "$TMP"

export KOSNICA_ML_DATA_RAW="$TMP/raw"
export KOSNICA_ML_SPLITS="$TMP/splits"
export KOSNICA_ML_RUNS="$TMP/runs"
export KOSNICA_ML_MODELS="$TMP/models"

echo "### 1/6 sinteticki dataset"
$PY tests/make_fake_dataset.py "$KOSNICA_ML_DATA_RAW"

echo; echo "### 2/6 EDA analiza dataseta"
$PY -m src.eda --out-dir "$TMP/report" | tail -5

echo; echo "### 3/6 podela podataka"
$PY -m src.split_data

echo; echo "### 4/6 trening: 3 arhitekture x 2 seme klasa, 1+1 epoha"
for arch in mobilenet_v3_large resnet50 efficientnet_b0; do
  for k in 6 3; do
    $PY -m src.train --arch "$arch" --classes "$k" \
        --epochs-head 1 --epochs-finetune 1 --batch-size 16 --workers 0
  done
done

echo; echo "### 5/6 evaluacija svih runova + zbirna tabela"
$PY -m src.evaluate --all

echo; echo "### 6/6 ONNX export sve tri arhitekture"
for run in mobilenet_v3_large_3cls resnet50_3cls efficientnet_b0_6cls; do
  $PY -m src.export_onnx --run "$run" --name "$run" 2>&1 | grep -v '^\[torch.onnx\]'
done

# Tezine MORAJU biti u samom .onnx fajlu — inace kopiranje modela u Docker
# daje model bez tezina.
if compgen -G "$KOSNICA_ML_MODELS/*.onnx.data" > /dev/null; then
  echo "GRESKA: pronadjen .onnx.data — model nije samostalan fajl."
  exit 1
fi
echo "Provera samostalnosti .onnx fajlova: OK"

echo; echo "SMOKE TEST PROSAO. Rezultati u $TMP (mogu se brisati)."
