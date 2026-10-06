#!/usr/bin/env bash
# Ceo eksperiment nad VarroaDataset-om (glavni eksperiment rada):
# preuzimanje -> 3 treninga pod istim uslovima -> evaluacija na test skupu ->
# provere da li model gleda grinju -> zbirna tabela.
#
# BeeImage eksperiment (prvi pokušaj, vidi NALAZI.md C-X) je u run_all.sh.
set -euo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "### Preuzimanje i statistika VarroaDataset-a"
$PY -u -m src.varroa_dataset | tail -25

echo; echo "### Trening: 3 arhitekture, isti uslovi"
for arch in mobilenet_v3_large resnet50 efficientnet_b0; do
  echo; echo "--- $arch ---"
  # već završen trening se preskače (config.json se piše tek na kraju treninga)
  if [ -f "runs/varroa_$arch/config.json" ]; then echo "već istreniran, preskačem"; continue; fi
  # Protokol izabran na val skupu (runs/tuning/, NALAZI.md C-Y):
  # 3 epohe samo glava, pa 15 epoha fine-tuning CELE mreže sa lr 1e-4, slika 224.
  $PY -u -m src.train --arch "$arch" --dataset varroa \
      --epochs-head 3 --epochs-finetune 15 --finetune-scope all --lr-finetune 1e-4
done

echo; echo "### Evaluacija na test skupu"
$PY -u -m src.evaluate --all --dataset varroa

echo; echo "### Da li model gleda grinju?"
$PY -u -m src.model_checks --all

echo; echo "### Test van skupa: terenske fotografije + BeeDataset"
$PY -u -m src.field_test

echo; echo "Gotovo. Rezultati: runs/varroa_*/ i runs/summary.csv"
