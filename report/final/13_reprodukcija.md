# Prilog: reprodukcija rezultata

Svi rezultati u izveštaju mogu se ponovo dobiti komandama ispod. Tabele u ovom
dokumentu generišu se automatski iz rezultata eksperimenta — nijedan broj nije
upisan ručno.

## Okruženje

| Komponenta | Verzija |
|---|---|
| Python | 3.12.15 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 |
| scikit-learn | 1.9.1 |
| onnxruntime | 1.30.0 |
| Hardver | Apple M5, PyTorch MPS backend |

```bash
cd KosnicaML
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Glavni eksperiment (VarroaDataset)

```bash
./run_varroa.sh
```

Skripta redom: preuzima VarroaDataset sa Zenodo-a (~1,2 GB), trenira tri
arhitekture, evaluira ih na test skupu, izvodi provere ponašanja i test van
skupa za trening. Trajanje na Apple M5: oko sat i po.

| Korak | Komanda | Rezultat |
|---|---|---|
| preuzimanje i statistika | `python -m src.varroa_dataset` | `data/varroa/` |
| trening | `python -m src.train --arch <arh> --dataset varroa` | `runs/varroa_<arh>/best.pt`, `history.csv` |
| evaluacija | `python -m src.evaluate --all --dataset varroa` | `metrics.json`, matrica konfuzije, krive učenja |
| provere ponašanja | `python -m src.model_checks --all` | `checks.json`, `gradcam.png`, `masking_examples.png` |
| test van skupa | `python -m src.field_test` | `runs/field_test.json` |
| izvoz u ONNX | `python -m src.export_onnx --run <run>` | `models/<ime>.onnx` + `.meta.json` |
| latencija | `python -m src.benchmark_onnx --pattern "varroa_*.onnx" --out latency_benchmark_varroa.csv` | CSV |
| provera servisa | `python -m service.verify_against_pytorch` | `models/verification.json` |
| izveštaj | `python -m src.make_final_report` | `report/izvestaj_final.{md,docx,pdf}` |

BeeDataset za test van skupa preuzima se sa `https://raspbee.de/BeeDataset_20201121.zip`
u `data/beedataset/`.

## Prvi eksperiment (BeeImage)

```bash
./run_all.sh                    # preuzimanje, 6 treninga, evaluacija
python -m src.shortcut_test     # test prečice
```

## Notebook-ovi

U folderu `notebooks/`, bez ponovnog treninga — čitaju gotove rezultate:

| Notebook | Sadržaj |
|---|---|
| `01_analiza_dataseta.ipynb` | BeeImage i VarroaDataset: klase, snimci, podela, primeri |
| `02_trening_i_poredjenje.ipynb` | transfer learning, krive učenja, poređenje tri arhitekture |
| `03_provera_modela.ipynb` | da li model gleda grinju, test van skupa, test kroz servis |

## Pokretanje aplikacije

```bash
# 1. ML servis
cd KosnicaML && ./.venv/bin/uvicorn service.main:app --port 8000
# 2. backend (http://localhost:5264)
cd KosnicaApi && dotnet watch
# 3. frontend (http://localhost:4200)
cd KosnicaUI && npm start
```
