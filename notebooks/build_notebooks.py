"""
Generiše notebook-ove u notebooks/ iz koda ispod, da ostanu usklađeni sa src/.
Notebook-ovi ne treniraju ništa — čitaju gotove rezultate iz runs/ i data/.

    ./.venv/bin/python notebooks/build_notebooks.py
    ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """\
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
pd.set_option("display.max_colwidth", 80)
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("napisan", name)


# ---------------------------------------------------------------------------
save("01_analiza_dataseta.ipynb", [
    md("""
# 1. Analiza skupova podataka

Ovaj notebook odgovara na prvo pitanje iz mejla profesorice: **koji skupovi
podataka postoje, koje klase sadrže i da li odgovaraju problemu**.

Upoređuju se dva skupa:
- **BeeImage** — korišćen u prvom eksperimentu; pokazalo se da nije pogodan
- **VarroaDataset** — korišćen u glavnom eksperimentu

Ključno pitanje za svaki skup: *da li oznaka opisuje pčelu na slici, ili nešto drugo?*
"""),
    code(SETUP),
    md("""
## 1.1 BeeImage: koliko različitih snimaka stoji iza svake klase?

Ime fajla (`038_112.png`) počinje id-jem video snimka iz kog je pčela isečena.
Ako klasa dolazi iz malog broja snimaka, model može da nauči snimak umesto bolesti.
"""),
    code("""
from src.dataset import load_split
bee = pd.concat([load_split(s, 3).assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
bee["snimak"] = bee["file"].str.split("_").str[0]

pregled = bee.groupby("label").agg(slika=("file", "size"), snimaka=("snimak", "nunique"))
pregled
"""),
    md("""
Klasa **varoa** ima 1.051 sliku, ali iz samo **5 snimaka**. Uz to, nasumična podela
je svaki od tih snimaka stavila i u trening i u test:
"""),
    code("""
v = bee[bee["label"] == "varoa_grinje"]
pd.crosstab(v["snimak"], v["split"])[["train", "val", "test"]]
"""),
    md("""
Test skup zato sadrži skoro iste frejmove kao trening — visoka tačnost na njemu ne
govori ništa o novim pčelama. Primeri slika po klasi (varoa ima prepoznatljivu
tirkiznu podlogu, a grinja se ne vidi):
"""),
    code("""
plt.figure(figsize=(12, 15)); plt.imshow(Image.open("report/figures/samples_per_class.png")); plt.axis("off");
"""),
    md("""
## 1.2 VarroaDataset

Svaka pčela je označena posebno: 0 = zdrava, 1 = vidljiva grinja, 3 = nedokumentovano
(grinja delimično skrivena; autori je broje kao zaraženu). Za svaku grinju postoji
ručno nacrtan pravougaonik — koristi se samo za provere, ne za trening.
"""),
    code("""
from src.varroa_dataset import parse_gt
df = parse_gt()
print("Ukupno slika:", len(df))
pd.crosstab(df["split"], df["label"], margins=True).loc[["train", "val", "test", "All"]]
"""),
    code("""
pd.crosstab(df["split"], df["raw_label"]).loc[["train", "val", "test"]]
"""),
    md("""
### Podela po snimku

Za razliku od BeeImage-a: **svaki snimak ima i zdrave i zaražene pčele**, i **nijedan
snimak nije u dva skupa**. Pozadina zato ne može da otkrije klasu, a test meri rad
na snimcima koje model nije video.
"""),
    code("""
snimci = df.groupby("video").agg(split=("split", "first"), slika=("target", "size"),
                                  udeo_varoe=("target", "mean")).sort_values("split")
print("Snimak u više skupova:", int((df.groupby("video")["split"].nunique() > 1).sum()))
snimci.round(2)
"""),
    md("### Veličina grinje na slici"),
    code("""
boxes = [b for bs in df["boxes"] for b in bs]
w = np.array([b[2] - b[0] for b in boxes]); h = np.array([b[3] - b[1] for b in boxes])
print(f"Grinja: širina medijana {np.median(w):.0f} px, visina {np.median(h):.0f} px (slika je 160×280)")
print("Broj grinja po zaraženoj pčeli:", df[df.n_mites > 0].n_mites.value_counts().sort_index().to_dict())
"""),
    md("### Primeri (crveni pravougaonik nacrtan iz `gt.csv`; na samim slikama ga nema)"),
    code("""
plt.figure(figsize=(14, 10)); plt.imshow(Image.open("report/figures/varroa_dataset_samples.png")); plt.axis("off");
"""),
    md("""
## 1.3 Zaključak

| | BeeImage | VarroaDataset |
|---|---|---|
| Oznaka | po snimku | po pčeli, vidljiva grinja |
| Pozadina po klasama | različita | ista |
| Snimaka po klasi varoe | 5 | svih 13 sadrži obe klase |
| Podela | nasumična (curenje) | po snimku |

VarroaDataset ispunjava sve uslove i koristi se u glavnom eksperimentu.
"""),
])

# ---------------------------------------------------------------------------
save("02_trening_i_poredjenje.ipynb", [
    md("""
# 2. Transfer learning i poređenje tri arhitekture

Ovaj notebook prikazuje **kako je trening urađen** i **upoređuje tri arhitekture**
(MobileNetV3-Large, ResNet50, EfficientNetB0) po metrikama koje je tražila
profesorica: accuracy, precision, recall, F1 i matrica konfuzije.

Sam trening se pokreće iz terminala (`./run_varroa.sh`, oko sat vremena); ovde se
čitaju njegovi rezultati iz `runs/varroa_*/`.

**Šta je naš kod, a šta biblioteka:** arhitekture mreža i njihove ImageNet težine
dolaze iz biblioteke `torchvision` i nisu menjane. Naš kod (`src/models.py`,
`src/train.py`) zamenjuje poslednji sloj, bira koji se slojevi treniraju, priprema
slike i vodi trening.
"""),
    code(SETUP),
    md("""
## 2.1 Transfer learning u tri poteza

1. učita se mreža istrenirana na ImageNet-u (1,28 miliona slika)
2. poslednji sloj (1.000 klasa) zameni se novim sa 2 izlaza: `zdrava`, `varoa`
3. faza 1: telo zamrznuto, trenira se samo nova glava (15 epoha);
   faza 2: odmrzne se poslednji blok tela i trenira sa 100× manjom stopom učenja (10 epoha)
"""),
    code("""
from src.models import build_model, freeze_backbone, unfreeze_last_block, count_parameters
rows = []
for arch in ["mobilenet_v3_large", "resnet50", "efficientnet_b0"]:
    model, _, _ = build_model(arch, 2, "letterbox")
    total = count_parameters(model)[0]
    freeze_backbone(model, arch); faza1 = count_parameters(model)[1]
    unfreeze_last_block(model, arch); faza2 = count_parameters(model)[1]
    rows.append({"arhitektura": arch, "ukupno": total, "trenira se u fazi 1": faza1, "u fazi 2": faza2})
pd.DataFrame(rows)
"""),
    md("""
## 2.2 Priprema slike i augmentacija

Slike su uspravne (160×280). Umesto isecanja centralnog kvadrata (koje bi odseklo glavu
i zadak) cela slika se smanji i dopuni do kvadrata — *letterbox*. Tokom treninga se
svaka slika nasumično rotira, preslika, pomera i menja joj se osvetljenje:
"""),
    code("""
plt.figure(figsize=(16, 3)); plt.imshow(Image.open("report/figures/varroa_augmentation.png")); plt.axis("off");
"""),
    md("## 2.3 Krive učenja\n\nVertikalna linija je prelaz u fine-tuning. Val skup su snimci koje model ne vidi u treningu."),
    code("""
RUNS = ROOT / "runs"
runs = sorted(p.name for p in RUNS.glob("varroa_*") if (p / "metrics.json").exists())
for run in runs:
    plt.figure(figsize=(13, 4.5)); plt.imshow(Image.open(RUNS / run / "learning_curves.png")); plt.axis("off"); plt.show()
"""),
    md("## 2.4 Poređenje na test skupu (3.408 slika, 2 snimka koje model nikad nije video)"),
    code("""
metrics = {r: json.loads((RUNS / r / "metrics.json").read_text()) for r in runs}
tabela = pd.DataFrame([{
    "arhitektura": m["arch_display"],
    "accuracy": m["accuracy"],
    "precision (macro)": m["macro_precision"],
    "recall (macro)": m["macro_recall"],
    "F1 (macro)": m["macro_f1"],
    "recall varoa": m["varroa_recall"],
    "ROC AUC": m.get("roc_auc"),
    "parametara (M)": round(m["total_params"] / 1e6, 1),
    "trening (min)": round(m["train_seconds"] / 60, 1),
} for m in metrics.values()]).sort_values("F1 (macro)", ascending=False)
tabela
"""),
    md("### Metrike po klasi"),
    code("""
for m in metrics.values():
    print(m["arch_display"])
    display(pd.DataFrame(m["per_class"]).T)
"""),
    md("""
### Matrice konfuzije

Red = stvarna klasa, kolona = šta je model rekao. Najskuplja greška za pčelara je
**zaražena pčela proglašena zdravom** (donji levi ugao).
"""),
    code("""
fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5))
for ax, run in zip(np.atleast_1d(axes), runs):
    ax.imshow(Image.open(RUNS / run / "confusion_matrix.png")); ax.axis("off")
"""),
    md("""
### Rezultat po test snimku

Test skup ima dva snimka sa vrlo različitim udelom zaraženih pčela (48% i 15%).
Ako bi model radio dobro samo na jednom, to bi ukazivalo na zavisnost od uslova snimanja.
"""),
    code("""
pd.DataFrame({m["arch_display"]: {v: d["accuracy"] for v, d in m["per_video"].items()}
              for m in metrics.values()})
"""),
    md("""
### Oznaka 1 (grinja jasno vidljiva) i oznaka 3 (delimično skrivena)

Recall na varoi, odvojeno po originalnoj oznaci autora. Očekuje se da je oznaka 3 teža.
"""),
    code("""
pd.DataFrame({m["arch_display"]: m["varroa_recall_by_raw_label"] for m in metrics.values()})
"""),
])

# ---------------------------------------------------------------------------
save("03_provera_modela.ipynb", [
    md("""
# 3. Da li model zaista gleda grinju?

Prvi model (BeeImage) imao je tačnost 97%, a nije prepoznavao bolest — prepoznavao je
pozadinu snimka. Zato se za novi model ne prihvata samo tačnost, nego se proverava
**na šta model gleda**, i kako radi na slikama koje nisu iz skupa za trening.
"""),
    code(SETUP + "\nRUNS = ROOT / 'runs'\nruns = sorted(p.name for p in RUNS.glob('varroa_*') if (p / 'checks.json').exists())"),
    md("""
## 3.1 Test 1 — brisanje grinje

Na svakoj zaraženoj test pčeli koju je model prepoznao, grinja se prekrije bojom
okoline. **Ako model gleda grinju, procena treba da pređe u „zdrava".**

Kontrola: isto toliko mesta se prekrije na drugom delu pčele. Tu procena treba da
ostane „varoa". Razlika između ova dva broja je dokaz.
"""),
    code("""
checks = {r: json.loads((RUNS / r / "checks.json").read_text()) for r in runs}
pd.DataFrame({c["arch_display"]: c["masking"] for c in checks.values()})
"""),
    code("""
for run in runs:
    plt.figure(figsize=(11, 10)); plt.imshow(Image.open(RUNS / run / "masking_examples.png")); plt.axis("off"); plt.show()
"""),
    md("""
## 3.2 Test 2 — samo pozadina

Modelu se daju samo uglovi slike (bez pčele). Pozadina je ista za obe klase, pa udeo
odgovora „varoa" mora biti sličan za uglove zdravih i zaraženih pčela.
Kod BeeImage modela, iz uglova je varoa „prepoznata" u 74% slučajeva.
"""),
    code("""
pd.DataFrame({c["arch_display"]: c["background"] for c in checks.values()})
"""),
    md("""
## 3.3 Test 3 — Grad-CAM

Grad-CAM pokazuje koji delovi slike su najviše doprineli odluci „varoa" (crveno = najviše).
Gornji red: grinja označena pravougaonikom iz `gt.csv`. Donji red: mapa pažnje modela.
„Pogodak" znači da je najjača tačka mape u pravougaoniku grinje.
"""),
    code("""
pd.DataFrame({c["arch_display"]: c["gradcam_pointing"] for c in checks.values()})
"""),
    code("""
for run in runs:
    plt.figure(figsize=(16, 7)); plt.imshow(Image.open(RUNS / run / "gradcam.png")); plt.axis("off"); plt.show()
"""),
    md("""
## 3.4 Test van skupa za trening

1. **BeeDataset** (7.490 slika) — druga kamera, druga zemlja, zelena podloga. Model ga nikad nije video.
2. **Terenske fotografije** (`data/field_photos/`) — fotografije pojedinačnih pčela sa bilo kojom
   pozadinom, uključujući onu koja je otkrila problem sa prvim modelom.
"""),
    code("""
field = json.loads((RUNS / "field_test.json").read_text())
pd.DataFrame({v["arch_display"]: v.get("beedataset", {}) for v in field.values()})
"""),
    code("""
rows = []
for v in field.values():
    for r in v.get("field", {}).get("po_slici", []):
        rows.append({"model": v["arch_display"], **r})
pd.DataFrame(rows).pivot(index=["fajl", "prava_klasa"], columns="model", values="p_varoa")
"""),
    md("""
## 3.5 Isti test kroz pravi servis

Fotografija koja je otkrila problem, poslata kroz isti kod koji koristi aplikacija
(`service/inference.py`, ONNX model koji je trenutno u produkciji):
"""),
    code("""
from service.inference import BeeHealthModel
served = BeeHealthModel(ROOT / "models" / "bee_health.onnx")
print("Model u servisu:", served.arch, "| klase:", served.classes)
for p in sorted((ROOT / "data" / "field_photos").glob("*/*")):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        print(f"{p.parent.name:6s} {p.name:40s} ->", served.predict(p.read_bytes()))
"""),
])
