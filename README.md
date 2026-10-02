# KosnicaML — klasifikacija zdravstvenog stanja pcela

ML deo diplomskog rada: transfer learning nad javnim datasetom fotografija pcela,
poredjenje tri arhitekture (MobileNetV3, ResNet50, EfficientNetB0) i integracija
najboljeg modela u informacioni sistem "Kosnica".

## Setup

```bash
cd KosnicaML
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # ocekuje se True
```

## Dataset

Potreban je Kaggle API token (`~/.kaggle/kaggle.json`, `chmod 600`).

```bash
python -m src.download_data
```

## Struktura

| Putanja | Sadrzaj |
|---|---|
| `data/raw/` | preuzet BeeImage dataset (gitignored) |
| `data/splits/` | train/val/test podela (komitovano — reproducibilnost) |
| `notebooks/01_eda.ipynb` | analiza dataseta, broj slika po klasi |
| `src/` | dataset, modeli, trening, evaluacija, ONNX export |
| `runs/` | rezultati po eksperimentu (metrike, grafikoni, history) |
| `models/` | izabrani model u ONNX formatu + metapodaci |
| `report/` | izvestaj za konsultacije |
