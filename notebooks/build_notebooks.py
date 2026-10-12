"""
Generiše notebook-ove u notebooks/ (KORAK 1–5). Grafici se crtaju u samim
notebook-ovima i čuvaju u grafici/. Ništa se ne trenira — čitaju se gotovi
rezultati iz runs/ (trening: ./run_varroa.sh).

    ./.venv/bin/python notebooks/build_notebooks.py
    ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = '''\
import sys, os, json, ast, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# src.evaluate pri importu prebacuje matplotlib u "Agg" (bez prikaza) — zato se
# učitava ovde, pa se prikaz u notebook-u vraća sa %matplotlib inline
import src.evaluate, src.model_checks
%matplotlib inline

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 12})
pd.set_option("display.max_colwidth", 60)

GRAFICI = ROOT / "grafici"; GRAFICI.mkdir(exist_ok=True)
RUNS = ROOT / "runs"

# Iste boje u svim graficima
BOJA_KLASE = {"zdrava": "#2ecc71", "varoa": "#e74c3c"}
BOJA_MODELA = {"MobileNetV3-Large": "#3498db", "ResNet50": "#9b59b6", "EfficientNetB0": "#f39c12"}
MODELI = ["MobileNetV3-Large", "ResNet50", "EfficientNetB0"]
RUN_ZA = {"MobileNetV3-Large": "varroa_mobilenet_v3_large", "ResNet50": "varroa_resnet50",
          "EfficientNetB0": "varroa_efficientnet_b0"}

def sacuvaj(ime):
    """Čuva trenutni grafik u grafici/<ime>.png (isti grafici idu u prezentaciju)."""
    plt.savefig(GRAFICI / f"{ime}.png", dpi=150, bbox_inches="tight")

def oznaci_stubice(ax, fmt="{:.3f}", rot=0, size=9):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center",
                        va="bottom", fontsize=size, rotation=rot, xytext=(0, 2), textcoords="offset points")

print("Biblioteke uspešno učitane. Koren projekta:", ROOT)
'''


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("napisan", name)


for old in HERE.glob("0*.ipynb"):
    old.unlink()

# =============================================================================
save("01_dataseti.ipynb", [
    md("""
# Prepoznavanje zdravstvenog stanja pčela sa fotografije — diplomski rad
## Informacioni sistem „Košnica“ — modul za analizu fotografije pčele
### KORAK 1: Izbor i analiza skupa podataka

**Problem:** na osnovu fotografije **pojedinačne pčele** proceniti da li je zaražena grinjom *Varroa destructor*.

Rad je urađen u pet koraka:

1. **Skupovi podataka** — koji postoje, koje klase imaju, da li odgovaraju problemu *(ovaj notebook)*
2. **Prvi eksperiment (BeeImage)** — zašto model sa 97% tačnosti ne radi
3. **Transfer learning** — kako se trenira, izbor protokola
4. **Evaluacija** — poređenje MobileNetV3, ResNet50 i EfficientNetB0
5. **Provera i izbor modela** — da li model gleda grinju, test van skupa, izbor

Ključno pitanje ovog koraka: **da li oznaka u skupu opisuje pčelu na slici — ili nešto drugo?**
"""),
    md("## 1. Uvoz biblioteka"),
    code(SETUP),
    md("""
## 2. Pregled javnih skupova podataka

| Skup | Slika | Klase | Oznaka po pčeli | Ista pozadina za sve klase | Podela po snimku |
|---|---|---|---|---|---|
| **BeeImage** (Kaggle) | 5.172 | 6 stanja košnice | ❌ po snimku | ❌ | ❌ |
| **VarroaDataset** (TU Wien) | 13.509 | zdrava / varoa | ✅ + ručno označena grinja | ✅ | ✅ zvanična |
| **BeeDataset** (BeeAlarmed) | 7.490 | varoa, polen, osa, hlađenje | ✅ | ✅ (zelena) | ❌ |
| Bilik i sar. 2021 | 600 | pčela, grinja | ✅ | ❌ | — (detekcija objekata) |
| Roboflow Universe | razno | grinja, pčela | ✅ | ❌ | — (detekcija objekata) |

U nastavku se detaljno porede prva dva.
"""),
    md("""
## 3. BeeImage — koliko snimaka stoji iza svake klase?

BeeImage je nastao isecanjem pčela iz **video snimaka** na ulazu u košnicu. Ime fajla
(`038_112.png`) počinje id-jem snimka. Ako klasa potiče iz malog broja snimaka, model može
naučiti *snimak* (pozadinu, osvetljenje) umesto bolesti.
"""),
    code('''
from src.dataset import load_split
bee = pd.concat([load_split(s, 3).assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
bee["snimak"] = bee["file"].str.split("_").str[0]

pregled = bee.groupby("label").agg(slika=("file", "size"), snimaka=("snimak", "nunique"))
pregled["slika po snimku"] = (pregled["slika"] / pregled["snimaka"]).round(0)
print("BeeImage — slike i snimci po klasi (3 grupisane klase)")
pregled
'''),
    code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
p = pregled.loc[["zdrava", "varoa_grinje", "drugi_problem"]]
boje = ["#2ecc71", "#e74c3c", "#95a5a6"]
axes[0].bar(p.index, p["slika"], color=boje); axes[0].set_title("Broj slika po klasi"); oznaci_stubice(axes[0], "{:.0f}")
axes[1].bar(p.index, p["snimaka"], color=boje); axes[1].set_title("Broj RAZLIČITIH snimaka po klasi"); oznaci_stubice(axes[1], "{:.0f}")
fig.suptitle("BeeImage: 1.051 slika varoe potiče iz svega 5 video snimaka", fontweight="bold")
plt.tight_layout(); sacuvaj("01a_beeimage_snimci_po_klasi"); plt.show()
'''),
    md("""
Nasumična podela (70/15/15) je **svaki od tih snimaka stavila i u trening i u test**. Uzastopni
frejmovi istog videa su skoro identični — test skup zato sadrži „iste“ slike kao trening:
"""),
    code('''
v = bee[bee["label"] == "varoa_grinje"]
tab = pd.crosstab(v["snimak"], v["split"])[["train", "val", "test"]]
plt.figure(figsize=(6, 3.4))
sns.heatmap(tab, annot=True, fmt="d", cmap="Reds", cbar=False, linewidths=0.5)
plt.title("BeeImage, klasa varoa: snimak × skup"); plt.ylabel("snimak (prefiks fajla)"); plt.xlabel("")
plt.tight_layout(); sacuvaj("01b_beeimage_curenje_snimaka"); plt.show()
'''),
    code('''
# Primeri po klasi — klasa varoe ima prepoznatljivu tirkiznu podlogu, a grinja se ne vidi
fig, axes = plt.subplots(3, 6, figsize=(13, 7))
for r, klasa in enumerate(["zdrava", "varoa_grinje", "drugi_problem"]):
    uzorak = bee[bee["label"] == klasa].sample(6, random_state=3)
    for c, path in enumerate(uzorak["path"]):
        axes[r, c].imshow(Image.open(path)); axes[r, c].axis("off")
    axes[r, 0].set_title(klasa, loc="left", fontsize=11)
fig.suptitle("BeeImage — primeri po klasi", fontweight="bold")
plt.tight_layout(); sacuvaj("01c_beeimage_primeri"); plt.show()
'''),
    md("""
**Zaključak za BeeImage:** oznaka opisuje **snimak**, ne pčelu. Svaki snimak pripada tačno jednoj
klasi, klasa varoe je praktično dva snimka, a na isečcima se grinja uglavnom ne vidi (srednja
veličina slike 61×61 piksela). Model može dobiti visoku tačnost prepoznajući pozadinu. Da se to
zaista dogodilo, pokazuje KORAK 2.
"""),
    md("""
## 4. VarroaDataset (izabran)

Schurischuster i Kampel, TU Wien, 2020 — Zenodo `10.5281/zenodo.4085044`, licenca CC BY 4.0.

Svaka pčela je označena posebno: **0 = zdrava**, **1 = vidljiva grinja**, **3 = nedokumentovano**
(grinja delimično skrivena; autori je računaju kao zaraženu). Za svaku grinju postoji ručno nacrtan
pravougaonik — koristi se **samo za proveru** modela, ne za trening.
"""),
    code('''
from src.varroa_dataset import parse_gt
df = parse_gt()
print("Ukupno slika:", len(df))
print("Snimaka:", df["video"].nunique(), "| snimak u više skupova:", int((df.groupby("video")["split"].nunique() > 1).sum()))
pd.crosstab(df["split"], df["raw_label"], margins=True).loc[["train", "val", "test", "All"]].rename(
    columns={0: "0 (zdrava)", 1: "1 (varoa)", 3: "3 (varoa, skrivena)"})
'''),
    code('''
fig, ax = plt.subplots(figsize=(8, 4))
t = pd.crosstab(df["split"], df["label"]).loc[["train", "val", "test"], ["zdrava", "varoa"]]
t.plot(kind="bar", stacked=True, color=[BOJA_KLASE["zdrava"], BOJA_KLASE["varoa"]], ax=ax, rot=0, width=0.6)
for i, (s, r) in enumerate(t.iterrows()):
    ax.text(i, r.sum() + 80, f"{r.sum():,}".replace(",", "."), ha="center", fontsize=9)
ax.set_title("VarroaDataset — slike po skupu i klasi"); ax.set_xlabel(""); ax.set_ylabel("slika")
plt.tight_layout(); sacuvaj("01d_varroa_raspodela"); plt.show()
'''),
    md("""
### Podela po snimku

Za razliku od BeeImage-a: **svaki snimak sadrži i zdrave i zaražene pčele**, a podelu po snimku su
dali autori — **nijedan snimak nije u dva skupa**. Pozadina zato ne može da otkrije klasu, a test
meri rad na snimcima koje model nikad nije video.
"""),
    code('''
sn = df.groupby("video").agg(skup=("split", "first"), slika=("target", "size"), udeo=("target", "mean"))
sn = sn.sort_values(["skup", "udeo"], key=lambda c: c.map({"train": 0, "val": 1, "test": 2}) if c.name == "skup" else c)
boja_skupa = {"train": "#3498db", "val": "#f39c12", "test": "#e74c3c"}
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(sn)), sn["udeo"] * 100, color=[boja_skupa[s] for s in sn["skup"]])
ax.set_yticks(range(len(sn))); ax.set_yticklabels([f"{v}  ({n})" for v, n in zip(sn.index, sn["slika"])], fontsize=8)
for i, u in enumerate(sn["udeo"]):
    ax.text(u * 100 + 1, i, f"{u:.0%}", va="center", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=s) for s, c in boja_skupa.items()], loc="lower right")
ax.set_xlabel("% zaraženih pčela u snimku"); ax.set_title("VarroaDataset — svaki snimak ima obe klase")
plt.tight_layout(); sacuvaj("01e_varroa_snimci"); plt.show()
'''),
    code('''
boxes = [b for bs in df["boxes"] for b in bs]
w = np.array([b[2] - b[0] for b in boxes]); h = np.array([b[3] - b[1] for b in boxes])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].hist(w, bins=30, color="#e74c3c", edgecolor="white"); axes[0].axvline(np.median(w), color="k", ls="--", label=f"medijana {np.median(w):.0f} px")
axes[0].set_title("Širina označene grinje (slika je 160 px široka)"); axes[0].legend()
n = df[df["n_mites"] > 0]["n_mites"].value_counts().sort_index()
axes[1].bar(n.index.astype(str), n.values, color="#e74c3c"); oznaci_stubice(axes[1], "{:.0f}")
axes[1].set_title("Broj grinja po zaraženoj pčeli"); axes[1].set_xlabel("grinja")
plt.tight_layout(); sacuvaj("01f_varroa_grinja"); plt.show()
'''),
    code('''
# Primeri (crveni pravougaonik je nacrtan iz gt.csv — na samim slikama ga nema)
fig, axes = plt.subplots(2, 8, figsize=(14, 6.4))
for r, (oznaka, naziv) in enumerate([(0, "zdrava"), (1, "varoa")]):
    uzorak = df[(df["raw_label"] == oznaka) & (df["split"] == "val")].sample(8, random_state=1)
    for c, (_, row) in enumerate(uzorak.iterrows()):
        im = Image.open(row["path"]).convert("RGB"); d = ImageDraw.Draw(im)
        for b in row["boxes"]: d.rectangle(b, outline=(255, 0, 0), width=2)
        axes[r, c].imshow(im); axes[r, c].axis("off")
    axes[r, 0].set_title(naziv, loc="left", fontsize=11, color=BOJA_KLASE[naziv])
fig.suptitle("VarroaDataset — primeri (grinja jasno vidljiva, ista podloga za obe klase)", fontweight="bold")
plt.tight_layout(); sacuvaj("01g_varroa_primeri"); plt.show()
'''),
    md("""
## 5. Zaključak KORAKA 1

| | BeeImage | VarroaDataset |
|---|---|---|
| Oznaka | po snimku | **po pčeli**, grinja vidljiva i označena |
| Pozadina po klasama | različita | **ista** |
| Snimci klase varoe | 5 (89% iz dva) | svih 13 snimaka ima obe klase |
| Podela | nasumična → isti snimak u train i test | **po snimku** (zvanična) |
| Veličina slike | ~61×61 px | 160×280 px, grinja ~30 px |

**Za trening je izabran VarroaDataset.** BeeDataset se koristi kao nezavisan test (druga kamera,
druga podloga).
"""),
])

# =============================================================================
save("02_prvi_eksperiment_beeimage.ipynb", [
    md("""
# Prepoznavanje zdravstvenog stanja pčela sa fotografije
### KORAK 2: Prvi eksperiment (BeeImage) — zašto model sa 97% tačnosti ne radi

Na BeeImage-u su trenirane sve tri arhitekture. Rezultati na test skupu su odlični. Ali ručni test u
aplikaciji je pokazao da model **ne prepoznaje bolest**. Ovaj notebook to dokazuje.
"""),
    md("## 1. Uvoz biblioteka"),
    code(SETUP),
    md("## 2. Rezultati na BeeImage test skupu (3 klase, 776 slika)"),
    code('''
bi = []
for p in sorted(RUNS.glob("*_3cls/metrics.json")):
    m = json.loads(p.read_text())
    bi.append({"Model": m["arch_display"], "Tačnost": m["accuracy"], "Preciznost": m["macro_precision"],
               "Odziv": m["macro_recall"], "F1": m["macro_f1"], "Odziv (varoa)": m["varroa_recall"]})
bi = pd.DataFrame(bi).set_index("Model").loc[MODELI]
print("BEEIMAGE — TEST SKUP (nasumična podela)"); print("=" * 50)
bi.round(4)
'''),
    code('''
ax = bi[["Tačnost", "Preciznost", "Odziv", "F1"]].plot(kind="bar", figsize=(10, 4.2), rot=0, width=0.75,
                                                      color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"])
ax.set_ylim(0.9, 1.005); ax.set_title("BeeImage — rezultati na test skupu izgledaju odlično (osa od 0,90)"); ax.set_xlabel("")
oznaci_stubice(ax, "{:.3f}", rot=90, size=8); ax.legend(loc="lower right")
plt.tight_layout(); sacuvaj("02a_beeimage_rezultati"); plt.show()
'''),
    md("""
## 3. Ručni test u aplikaciji

Fotografija pčele na saću sa **dve jasno vidljive grinje**, propuštena kroz BeeImage model
(isti ONNX fajl koji je tada bio u aplikaciji):
"""),
    code('''
from service.inference import BeeHealthModel
stari = BeeHealthModel(ROOT / "models" / "beeimage" / "mobilenet_v3_large_3cls.onnx")
foto = ROOT / "data" / "field_photos" / "varoa" / "makro_pcela_na_sacu_2_grinje.jpg"
r = stari.predict(foto.read_bytes())

fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
axes[0].imshow(Image.open(foto)); axes[0].axis("off"); axes[0].set_title("Dve grinje jasno vidljive")
k = list(r["all_probs"]); v = list(r["all_probs"].values())
axes[1].barh(k, v, color=["#2ecc71", "#e74c3c", "#95a5a6"]); axes[1].set_xlim(0, 1)
for i, x in enumerate(v): axes[1].text(x + 0.01, i, f"{x:.2%}", va="center")
axes[1].set_title("BeeImage model: „zdrava“")
plt.tight_layout(); sacuvaj("02b_beeimage_promasaj"); plt.show()
print(r)
'''),
    md("""
## 4. Test prečice (*shortcut test*)

**Hipoteza:** model je naučio pozadinu snimka, a ne izgled pčele.

**Test:** modelu se daju **samo četiri ugla** slike (25% širine i visine). Pčela je u centru isečka,
pa su u uglovima uglavnom pozadina i podloga. Kad bi model gledao pčelu, iz uglova bi pogađao na
nivou slučaja — **33%** za tri klase.
"""),
    code('''
import io
def png(img):
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()

test = load_split("test", 3) if "load_split" in dir() else None
from src.dataset import load_split
test = load_split("test", 3)
rez = []
for klasa in stari.classes:
    sub = test[test["label"] == klasa]
    cela = uglovi = 0.0
    for path in sub["path"]:
        img = Image.open(path).convert("RGB"); w, h = img.size; qw, qh = w // 4, h // 4
        cela += stari.predict(png(img))["label"] == klasa
        kutije = [(0, 0, qw, qh), (w - qw, 0, w, qh), (0, h - qh, qw, h), (w - qw, h - qh, w, h)]
        uglovi += sum(stari.predict(png(img.crop(b)))["label"] == klasa for b in kutije) / 4
    rez.append({"klasa": klasa, "slika": len(sub), "cela slika": cela / len(sub), "samo uglovi": uglovi / len(sub)})
rez = pd.DataFrame(rez).set_index("klasa"); rez.round(3)
'''),
    code('''
ax = rez[["cela slika", "samo uglovi"]].plot(kind="bar", figsize=(9, 4.2), rot=0, color=["#3498db", "#e74c3c"], width=0.7)
ax.axhline(1 / 3, color="k", ls="--", lw=1); ax.text(2.42, 0.35, "slučajno (33%)", fontsize=9, ha="right")
oznaci_stubice(ax, "{:.0%}"); ax.set_ylim(0, 1.1); ax.set_xlabel("")
ax.set_title("Tačnost BeeImage modela: cela slika vs. samo uglovi (bez pčele)")
plt.tight_layout(); sacuvaj("02c_test_precice"); plt.show()
'''),
    code('''
# Kako izgleda ono što model dobija u testu: uglovi slika iz klase varoe
uz = test[test["label"] == "varoa_grinje"].sample(4, random_state=0)
fig, axes = plt.subplots(2, 4, figsize=(11, 5))
for c, path in enumerate(uz["path"]):
    img = Image.open(path).convert("RGB"); w, h = img.size
    axes[0, c].imshow(img); axes[0, c].axis("off")
    ugao = img.crop((0, 0, w // 4, h // 4))
    axes[1, c].imshow(ugao); axes[1, c].axis("off")
    axes[1, c].set_title(f"ugao → {stari.predict(png(ugao))['label']}", fontsize=9)
axes[0, 0].set_title("cela slika", loc="left"); fig.suptitle("Iz komadića pozadine model „prepoznaje“ varou", fontweight="bold")
plt.tight_layout(); sacuvaj("02d_uglovi_primeri"); plt.show()
'''),
    md("""
## 5. Zaključak KORAKA 2

Iz komadića pozadine **na kome pčele nema**, model u oko **74%** slučajeva „prepoznaje“ varou.
Tačnost od 97% je tačan broj, ali meri **prepoznavanje snimka**, a ne bolesti. Ova pojava se u
literaturi zove *shortcut learning*.

**Pouke ugrađene u glavni eksperiment:**
1. oznaka **po pčeli**, na osnovu vidljivog obeležja
2. **ista pozadina** za sve klase
3. podela **po snimku** — test sadrži snimke koje model nije video
4. visoka tačnost nije dovoljna — treba **proveriti šta model gleda** (KORAK 5)
"""),
])

# =============================================================================
save("03_transfer_learning_i_trening.ipynb", [
    md("""
# Prepoznavanje zdravstvenog stanja pčela sa fotografije
### KORAK 3: Transfer learning i trening na VarroaDataset-u

Tri arhitekture — **MobileNetV3-Large, ResNet50, EfficientNetB0** — prethodno istrenirane na ImageNet-u
(1,28 miliona slika) prilagođavaju se zadatku *zdrava / varoa* pod **istim uslovima**.

Trening se pokreće iz terminala (`./run_varroa.sh`, oko 1,5 h na Apple M5); ovde se prikazuje postupak i
čitaju rezultati iz `runs/`.

**Šta je biblioteka, a šta naš kod:** arhitekture i ImageNet težine dolaze iz `torchvision` i nisu menjane.
Naš kod (`src/models.py`, `src/train.py`) menja poslednji sloj, bira šta se trenira, priprema slike i vodi trening.
"""),
    md("## 1. Uvoz biblioteka"),
    code(SETUP),
    md("""
## 2. Transfer learning u tri poteza

1. učita se mreža istrenirana na ImageNet-u
2. poslednji sloj (1.000 klasa) zameni se novim sa **2 izlaza** — `zdrava`, `varoa`
3. **faza 1:** telo zamrznuto, trenira se samo nova glava (3 epohe, lr 10⁻³) →
   **faza 2:** odmrzne se **cela mreža** i trenira sa 10× manjom stopom učenja (15 epoha, lr 10⁻⁴)
"""),
    code('''
from src.models import build_model, freeze_backbone, unfreeze_all, count_parameters, display_name
param = []
for arch in ["mobilenet_v3_large", "resnet50", "efficientnet_b0"]:
    model, _, _ = build_model(arch, 2, "letterbox")
    ukupno = count_parameters(model)[0]
    freeze_backbone(model, arch); faza1 = count_parameters(model)[1]
    param.append({"Model": display_name(arch), "ukupno parametara": ukupno, "trenira se u fazi 1": faza1, "u fazi 2": ukupno})
param = pd.DataFrame(param).set_index("Model"); param
'''),
    code('''
ax = (param[["ukupno parametara", "trenira se u fazi 1"]] / 1e6).plot(kind="bar", figsize=(9, 4), rot=0,
                                                                   color=["#34495e", "#1abc9c"], width=0.7)
oznaci_stubice(ax, "{:.2f} M"); ax.set_ylabel("miliona parametara"); ax.set_xlabel("")
ax.set_title("Veličina modela i broj parametara koji se treniraju u fazi 1")
plt.tight_layout(); sacuvaj("03a_parametri"); plt.show()
'''),
    md("""
## 3. Priprema slike: zašto *letterbox*, a ne standardno isecanje

Standardna priprema (`weights.transforms()`) smanji kraću stranu na 232 px i iseče centralni kvadrat 224×224.
Slike u VarroaDataset-u su **uspravne (160×280)** — isecanje bi odbacilo glavu i zadak pčele, gde je grinja često.
**Letterbox** smanji celu sliku i dopuni je do kvadrata. Ista funkcija (`service/preprocessing.py`) radi i u servisu.
"""),
    code('''
from src.varroa_dataset import parse_gt
from service.preprocessing import letterbox
gt = parse_gt()
row = gt[(gt["n_mites"] > 0) & (gt["split"] == "val")].iloc[5]
img = Image.open(row["path"]).convert("RGB"); d = ImageDraw.Draw(img)
for b in row["boxes"]: d.rectangle(b, outline=(255, 0, 0), width=2)
w, h = img.size; s = 232 / w; rs = img.resize((232, round(h * s)))
top = (rs.height - 224) // 2; centar = rs.crop((4, top, 228, top + 224))

fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
axes[0].imshow(img); axes[0].set_title("original 160×280")
axes[1].imshow(centar); axes[1].set_title("standardno isecanje 224×224\\n(odsečeno ~45% visine)", color="#c0392b")
axes[2].imshow(letterbox(img, 224)); axes[2].set_title("letterbox 224×224\\n(cela pčela)", color="#27ae60")
for a in axes: a.axis("off")
plt.tight_layout(); sacuvaj("03b_letterbox"); plt.show()
'''),
    md("""
## 4. Augmentacija

Tokom treninga svaka slika se pri svakom prolazu nasumično menja: rotacija u svim smerovima (pčela je
snimljena odozgo), preslikavanje, pomeranje, promena razmere, osvetljenja, kontrasta, boje i blago
zamućenje. Nijansa se menja vrlo malo — crvenkasta boja grinje je signal.
"""),
    code('''
import torch
_, train_tf, eval_tf = build_model("mobilenet_v3_large", 2, "letterbox")
torch.manual_seed(0)
orig = Image.open(row["path"]).convert("RGB")
mean = torch.tensor(eval_tf.mean).view(3, 1, 1); std = torch.tensor(eval_tf.std).view(3, 1, 1)
fig, axes = plt.subplots(1, 8, figsize=(16, 2.6))
for i, ax in enumerate(axes):
    x = eval_tf(orig) if i == 0 else train_tf(orig)
    ax.imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0)); ax.axis("off")
    ax.set_title("original" if i == 0 else f"varijanta {i}", fontsize=9)
plt.tight_layout(); sacuvaj("03c_augmentacija"); plt.show()
'''),
    md("""
## 5. Izbor protokola treninga (samo na validacionom skupu)

Prvi pokušaj (**A**) koristio je protokol iz BeeImage eksperimenta: fine-tuning samo **poslednjeg bloka**.
Model nije naučio zadatak. Varijante su upoređene **isključivo na validacionom skupu** (drugi snimci),
na MobileNetV3 kao najbržoj mreži; izabrani protokol je zatim primenjen na sve tri arhitekture.
Test skup nije korišćen.
"""),
    code('''
tun = []
fig, ax = plt.subplots(figsize=(10, 4.5))
boje = {"A": "#95a5a6", "B": "#27ae60", "C": "#2980b9"}
for d in sorted((RUNS / "tuning").iterdir()):
    c = json.loads((d / "config.json").read_text()); hst = pd.read_csv(d / "history.csv")
    k = d.name[0]
    opis = f"{k}: {'cela mreža' if c.get('finetune_scope') == 'all' else 'poslednji blok'}, {c.get('img_size', 224)} px"
    ax.plot(range(1, len(hst) + 1), hst["val_macro_f1"], marker="o", ms=3, label=opis, color=boje[k])
    tun.append({"varijanta": opis, "train tačnost (kraj)": hst["train_acc"].iloc[-1],
                "najbolji val F1": c["best_val_macro_f1"], "trening (min)": round(c["train_seconds"] / 60, 1)})
ax.set_xlabel("epoha (ukupno)"); ax.set_ylabel("val macro F1"); ax.set_ylim(0.3, 1)
ax.set_title("Izbor protokola: fine-tuning cele mreže je presudan"); ax.legend()
plt.tight_layout(); sacuvaj("03d_izbor_protokola"); plt.show()
pd.DataFrame(tun).set_index("varijanta").round(4)
'''),
    md("""
**Zaključak:** kada se odmrzne cela mreža, validacioni F1 skače sa **0,665 na 0,934**. Zamrznuti
slojevi, naučeni na ImageNet-u, ne izdvajaju sitno obeležje kao što je grinja. Veća slika (C, 320 px)
daje još ~0,01 — manje od oscilacija između epoha — a 4,7× je sporija. **Izabrana je varijanta B.**
"""),
    md("## 6. Krive učenja — sve tri arhitekture, isti protokol"),
    code('''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for m in MODELI:
    h = pd.read_csv(RUNS / RUN_ZA[m] / "history.csv"); x = range(1, len(h) + 1)
    axes[0].plot(x, h["train_loss"], color=BOJA_MODELA[m], ls="--", alpha=0.7)
    axes[0].plot(x, h["val_loss"], color=BOJA_MODELA[m], label=m)
    axes[1].plot(x, h["val_macro_f1"], color=BOJA_MODELA[m], marker="o", ms=3, label=m)
for a in axes:
    a.axvline(3.5, color="gray", ls=":"); a.set_xlabel("epoha"); a.legend()
axes[0].set_title("Loss (puna linija = val, isprekidana = train)"); axes[1].set_title("Validacioni macro F1")
axes[1].text(3.7, 0.45, "fine-tuning →", color="gray"); axes[1].set_ylim(0.4, 1)
plt.tight_layout(); sacuvaj("03e_krive_ucenja"); plt.show()
cfg = {m: json.loads((RUNS / RUN_ZA[m] / "config.json").read_text()) for m in MODELI}
pd.DataFrame({m: {"najbolji val F1": c["best_val_macro_f1"], "epoha": f"{c['best_stage']} {c['best_epoch']}",
                  "trening (min)": round(c["train_seconds"] / 60, 1)} for m, c in cfg.items()}).T
'''),
    md("""
**Zaključak KORAKA 3:** sve tri mreže uče stabilno; val loss prati train loss (nema izraženog
preprilagođavanja). Najbolji checkpoint se bira po val macro F1 — test skup tek u KORAKU 4.
"""),
])

# =============================================================================
save("04_evaluacija_i_poredjenje.ipynb", [
    md("""
# Prepoznavanje zdravstvenog stanja pčela sa fotografije
### KORAK 4: Evaluacija i poređenje tri arhitekture

Test skup: **3.408 slika iz 2 snimka koje modeli nikad nisu videli** (942 zaražene, 2.466 zdravih).
Metrike koje je tražila profesorica: **tačnost, preciznost, odziv (recall), F1 i matrica konfuzije**.
"""),
    md("## 1. Uvoz biblioteka"),
    code(SETUP),
    md("## 2. Predikcije modela na test skupu"),
    code('''
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)
from src.predictions import predictions
pred = {m: predictions(RUN_ZA[m], "test") for m in MODELI}
print({m: len(p) for m, p in pred.items()})
'''),
    md("## 3. Tabela rezultata"),
    code('''
rez = []
for m, p in pred.items():
    y, yp = p["true"], p["pred"]
    rez.append({"Model": m, "Tačnost": accuracy_score(y, yp),
                "Preciznost": precision_score(y, yp, average="macro"), "Odziv": recall_score(y, yp, average="macro"),
                "F1": f1_score(y, yp, average="macro"), "Odziv (varoa)": recall_score(y, yp),
                "Preciznost (varoa)": precision_score(y, yp), "ROC AUC": roc_auc_score(y, p["p_varoa"])})
rez = pd.DataFrame(rez).set_index("Model").sort_values("F1", ascending=False)
print("VARROADATASET — TEST SKUP (sortirano po F1)"); print("=" * 55)
rez.round(4)
'''),
    code('''
ax = rez.loc[MODELI, ["Tačnost", "Preciznost", "Odziv", "F1"]].plot(kind="bar", figsize=(11, 4.6), rot=0, width=0.78,
                                                                   color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"])
ax.set_ylim(0.8, 0.96); oznaci_stubice(ax, "{:.3f}", rot=90, size=8); ax.set_xlabel("")
ax.set_title("Poređenje arhitektura na test skupu (macro proseci; osa počinje od 0,80)"); ax.legend(loc="upper right", ncol=4)
plt.tight_layout(); sacuvaj("04a_poredjenje_metrika"); plt.show()
'''),
    md("""
## 4. Matrice konfuzije

Red = stvarna klasa, kolona = odluka modela. Najskuplja greška za pčelara je **propuštena zaraza**
(donji levi kvadrat); gornji desni su **lažne uzbune**.
"""),
    code('''
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, m in zip(axes, MODELI):
    cm = confusion_matrix(pred[m]["true"], pred[m]["pred"]); pc = cm / cm.sum(axis=1, keepdims=True)
    ann = np.array([[f"{cm[i, j]}\\n({pc[i, j]:.1%})" for j in range(2)] for i in range(2)])
    sns.heatmap(pc, annot=ann, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=["zdrava", "varoa"], yticklabels=["zdrava", "varoa"], annot_kws={"size": 11})
    ax.set_title(m, color=BOJA_MODELA[m]); ax.set_xlabel("predviđeno"); ax.set_ylabel("stvarno")
plt.tight_layout(); sacuvaj("04b_matrice_konfuzije"); plt.show()
'''),
    md("""
## 5. ROC krive

ROC kriva pokazuje odnos otkrivenih zaraza i lažnih uzbuna za **svaki mogući prag** odluke (ne samo 0,5).
Površina ispod krive (AUC) je 1,0 za savršen model i 0,5 za nasumično pogađanje.
"""),
    code('''
plt.figure(figsize=(6.5, 5.5))
for m in MODELI:
    fpr, tpr, _ = roc_curve(pred[m]["true"], pred[m]["p_varoa"])
    plt.plot(fpr, tpr, color=BOJA_MODELA[m], lw=2, label=f"{m} (AUC = {roc_auc_score(pred[m]['true'], pred[m]['p_varoa']):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="nasumično")
plt.xlabel("lažne uzbune (FPR)"); plt.ylabel("otkrivene zaraze (TPR, recall)")
plt.title("ROC krive — test skup"); plt.legend(loc="lower right")
plt.tight_layout(); sacuvaj("04c_roc_krive"); plt.show()
'''),
    md("## 6. Rezultat po test snimku i po vidljivosti grinje"),
    code('''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
po_snimku = pd.DataFrame({m: pred[m].groupby("video").apply(lambda g: (g["true"] == g["pred"]).mean()) for m in MODELI})
udeo = pred[MODELI[0]].groupby("video")["true"].mean()
po_snimku.index = [f"{v}\\n({u:.0%} zaraženih)" for v, u in zip(po_snimku.index, udeo)]
po_snimku.plot(kind="bar", ax=axes[0], rot=0, color=[BOJA_MODELA[m] for m in MODELI], width=0.75)
axes[0].set_ylim(0.8, 1); oznaci_stubice(axes[0], "{:.3f}", size=8); axes[0].set_title("Tačnost po test snimku")

po_oznaci = pd.DataFrame({m: pred[m][pred[m]["true"] == 1].groupby("raw_label")["pred"].mean() for m in MODELI})
po_oznaci.index = ["oznaka 1\\n(grinja vidljiva)", "oznaka 3\\n(delimično skrivena)"]
po_oznaci.plot(kind="bar", ax=axes[1], rot=0, color=[BOJA_MODELA[m] for m in MODELI], width=0.75)
axes[1].set_ylim(0, 1.05); oznaci_stubice(axes[1], "{:.2f}", size=8); axes[1].set_title("Odziv na varoi po vidljivosti grinje")
plt.tight_layout(); sacuvaj("04d_po_snimku_i_oznaci"); plt.show()
'''),
    md("""
**Zaključak:** rezultat je ujednačen na oba test snimka — modeli ne zavise od uslova jednog snimka.
Kada je grinja jasno vidljiva (oznaka 1), sva tri modela je pronalaze u ~93–95% slučajeva; kada je
delimično skrivena (oznaka 3), u ~55–62% — tu leži najveći deo propuštenih zaraza.
"""),
    md("""
## 7. Analiza grešaka — ResNet50

*Gde model najviše greši i zašto?*
"""),
    code('''
p = pred["ResNet50"].copy()
p["boxes"] = p["boxes"].apply(ast.literal_eval)
fn = p[(p["true"] == 1) & (p["pred"] == 0)]; fp = p[(p["true"] == 0) & (p["pred"] == 1)]
print(f"Propuštene zaraze (FN): {len(fn)} od {int(p['true'].sum())}  —  od toga oznaka 3 (skrivena grinja): {int((fn['raw_label'] == 3).sum())}")
print(f"Lažne uzbune (FP):      {len(fp)} od {int((p['true'] == 0).sum())}")

def mreza(frame, naslov, ime, n=8):
    uz = frame.sort_values("p_varoa", ascending=naslov.startswith("Propuštene")).head(n)
    fig, axes = plt.subplots(1, n, figsize=(16, 3.9))
    for ax, (_, r) in zip(axes, uz.iterrows()):
        im = Image.open(r["path"]).convert("RGB"); d = ImageDraw.Draw(im)
        for b in r["boxes"]: d.rectangle(b, outline=(255, 0, 0), width=2)
        ax.imshow(im); ax.axis("off"); ax.set_title(f"P(varoa)={r['p_varoa']:.2f}\\noznaka {r['raw_label']}", fontsize=8)
    fig.suptitle(naslov, fontweight="bold"); plt.tight_layout(); sacuvaj(ime); plt.show()

mreza(fn, "Propuštene zaraze — najsigurnije greške (crveno: grinja iz gt.csv)", "04e_greske_propustene")
mreza(fp, "Lažne uzbune — zdrave pčele koje je model proglasio zaraženim", "04f_greske_lazne_uzbune")
'''),
    code('''
plt.figure(figsize=(9, 4))
plt.hist(p[p["true"] == 0]["p_varoa"], bins=40, alpha=0.75, color=BOJA_KLASE["zdrava"], label="stvarno zdrave")
plt.hist(p[p["true"] == 1]["p_varoa"], bins=40, alpha=0.75, color=BOJA_KLASE["varoa"], label="stvarno zaražene")
plt.axvline(0.5, color="k", ls="--", label="prag 0,5"); plt.yscale("log")
plt.xlabel("P(varoa) koju daje ResNet50"); plt.ylabel("broj slika (log)"); plt.legend()
plt.title("Koliko je model siguran — raspodela verovatnoća po stvarnoj klasi")
plt.tight_layout(); sacuvaj("04g_raspodela_verovatnoca"); plt.show()
'''),
    md("""
## 8. Zaključak KORAKA 4

- **ResNet50** je najbolji po svim agregatnim metrikama (tačnost 0,931, F1 0,913, AUC 0,975) i ima
  najmanje lažnih uzbuna.
- **MobileNetV3** ima najveći odziv na varoi (0,876), uz više lažnih uzbuna.
- Razlike između arhitektura su ~1 procentni poen — svaka je trenirana jednom, pa ih ne treba precenjivati.
- Najveći izvor grešaka su **delimično skrivene grinje** (oznaka 3) i odsjaji/tamna mesta koja liče na grinju.

Da li modeli zaista gledaju grinju — KORAK 5.
"""),
])

# =============================================================================
save("05_provera_i_izbor_modela.ipynb", [
    md("""
# Prepoznavanje zdravstvenog stanja pčela sa fotografije
### KORAK 5: Da li model gleda grinju? Test van skupa i izbor modela

BeeImage model je imao 97% i nije gledao bolest. Zato se za nove modele ne prihvata samo tačnost —
proverava se **na šta model gleda** i kako radi **na slikama drugačijeg porekla**.
"""),
    md("## 1. Uvoz biblioteka"),
    code(SETUP + '''
checks = {m: json.loads((RUNS / RUN_ZA[m] / "checks.json").read_text()) for m in MODELI}
'''),
    md("""
## 2. Provera 1 — brisanje grinje (uzročni test)

Na svakoj zaraženoj test pčeli koju je model prepoznao, grinja se **prekrije bojom okoline**.
Ako model gleda grinju → procena prelazi u „zdrava“.
**Kontrola:** isto toliko mesta se prekrije na **drugom delu pčele** → procena treba da ostane „varoa“.
"""),
    code('''
mk = pd.DataFrame({m: {"original": c["masking"]["prepoznato_kao_varoa_original"] / c["masking"]["prepoznato_kao_varoa_original"],
                       "grinja prekrivena": c["masking"]["ostalo_varoa_kad_se_grinja_prekrije"],
                       "drugo mesto prekriveno": c["masking"]["ostalo_varoa_kad_se_prekrije_drugo_mesto"]}
                   for m, c in checks.items()}).T
ax = mk.plot(kind="bar", figsize=(10, 4.4), rot=0, color=["#7f8c8d", "#2ecc71", "#e74c3c"], width=0.75)
oznaci_stubice(ax, "{:.0%}"); ax.set_ylim(0, 1.15); ax.set_xlabel("")
ax.set_ylabel("udeo koji je i dalje „varoa“"); ax.legend(loc="upper center", ncol=3)
ax.set_title("Kad se ukloni grinja, model menja mišljenje; kad se ukloni bilo šta drugo — ne menja")
plt.tight_layout(); sacuvaj("05a_brisanje_grinje"); plt.show()
'''),
    code('''
plt.figure(figsize=(10, 9)); plt.imshow(Image.open(RUNS / "varroa_resnet50" / "masking_examples.png")); plt.axis("off")
plt.tight_layout(); sacuvaj("05b_brisanje_primeri"); plt.show()
'''),
    md("""
## 3. Provera 2 — samo pozadina

Modelu se daju samo uglovi slike (bez pčele). Kod BeeImage modela iz uglova je „prepoznata“ varoa u 74%
slučajeva. Novi modeli bi iz pozadine trebalo da kažu „varoa“ podjednako retko za zdrave i zaražene pčele.
"""),
    code('''
bg = pd.DataFrame({m: {"uglovi zdravih": c["background"]["udeo_varoa_uglovi_zdravih"],
                       "uglovi zaraženih": c["background"]["udeo_varoa_uglovi_zarazenih"]} for m, c in checks.items()}).T
bg.loc["BeeImage model\\n(prvi eksperiment)"] = [np.nan, 0.739]
ax = bg.plot(kind="bar", figsize=(10, 4.2), rot=0, color=["#2ecc71", "#e74c3c"], width=0.7)
oznaci_stubice(ax, "{:.1%}"); ax.set_ylim(0, 0.9); ax.set_xlabel("")
ax.set_title("Koliko često model kaže „varoa“ gledajući SAMO pozadinu")
plt.tight_layout(); sacuvaj("05c_samo_pozadina"); plt.show()
'''),
    md("""
## 4. Provera 3 — Grad-CAM (gde model gleda)

Grad-CAM boji delove slike koji su najviše doprineli odluci „varoa“ (crveno = najviše).
Gornji red: grinja iz `gt.csv`; donji red: mapa pažnje.
"""),
    code('''
pg = pd.Series({m: c["gradcam_pointing"]["pogodak_u_grinju"] for m, c in checks.items()})
ax = pg.plot(kind="bar", figsize=(7, 3.6), rot=0, color=[BOJA_MODELA[m] for m in pg.index])
oznaci_stubice(ax, "{:.0%}"); ax.set_ylim(0, 1.05); ax.set_title("Najjača tačka Grad-CAM mape je na grinji")
plt.tight_layout(); sacuvaj("05d_gradcam_pogoci"); plt.show()
for m in ["MobileNetV3-Large", "ResNet50"]:
    plt.figure(figsize=(16, 6.6)); plt.imshow(Image.open(RUNS / RUN_ZA[m] / "gradcam.png")); plt.axis("off"); plt.show()
'''),
    md("""
**Zaključak provera:** odluka sva tri modela **zavisi od grinje** (uklanjanje grinje ruši procenu sa ~87%
na 3–4%, uklanjanje drugog dela pčele ne). ResNet50 ima širu Grad-CAM mapu (obuhvata i okolni deo tela),
ali uzročni test — brisanje grinje — je presudan.
"""),
    md("""
## 5. Test van skupa — BeeDataset (druga kamera, zelena podloga)

7.490 slika koje model nikad nije video, snimljenih potpuno drugom opremom.
"""),
    code('''
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, accuracy_score
from src.predictions import predictions
bd = {m: predictions(RUN_ZA[m], "beedataset") for m in MODELI}
tab = pd.DataFrame({m: {"Tačnost": accuracy_score(p["true"], p["pred"]), "Odziv (varoa)": recall_score(p["true"], p["pred"]),
                        "Preciznost (varoa)": precision_score(p["true"], p["pred"]), "ROC AUC": roc_auc_score(p["true"], p["p_varoa"])}
                    for m, p in bd.items()}).T
tab.round(4)
'''),
    code('''
uz = bd["ResNet50"].groupby("true").sample(4, random_state=2)
fig, axes = plt.subplots(1, 8, figsize=(14, 3.8))
for ax, (_, r) in zip(axes, uz.iterrows()):
    ax.imshow(Image.open(r["path"])); ax.axis("off")
    ax.set_title(("varoa" if r["true"] else "zdrava") + f"\\nP={r['p_varoa']:.2f}", fontsize=9,
                 color=BOJA_KLASE["varoa" if r["true"] else "zdrava"])
fig.suptitle("BeeDataset — primeri sa P(varoa) koju daje ResNet50", fontweight="bold")
plt.tight_layout(); sacuvaj("05e_beedataset_primeri"); plt.show()
'''),
    code('''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for m in MODELI:
    p = bd[m]; fpr, tpr, _ = roc_curve(p["true"], p["p_varoa"])
    axes[0].plot(fpr, tpr, color=BOJA_MODELA[m], lw=2, label=f"{m} (AUC {roc_auc_score(p['true'], p['p_varoa']):.3f})")
    pragovi = np.linspace(0.05, 0.95, 19)
    axes[1].plot(pragovi, [((p["p_varoa"] >= t) & (p["true"] == 1)).sum() / p["true"].sum() for t in pragovi],
                 color=BOJA_MODELA[m], marker="o", ms=3, label=m)
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("lažne uzbune (FPR)"); axes[0].set_ylabel("otkrivene zaraze")
axes[0].set_title("BeeDataset — ROC krive"); axes[0].legend(loc="lower right")
axes[1].axvline(0.5, color="k", ls="--", lw=1); axes[1].set_xlabel("prag odluke"); axes[1].set_ylabel("odziv na varoi")
axes[1].set_title("BeeDataset — odziv na varoi u zavisnosti od praga"); axes[1].legend()
plt.tight_layout(); sacuvaj("05f_beedataset_roc_prag"); plt.show()
'''),
    md("""
**Zaključak:** modeli dobro **razdvajaju** zaražene od zdravih i na drugoj kameri (AUC 0,87–0,95), ali su
**previše oprezni**: na pragu 0,5 hvataju samo 20–48% zaraza. Spuštanjem praga odziv brzo raste — to je
jeftino poboljšanje bez ponovnog treninga (desni grafik).
"""),
    md("## 6. Terenske fotografije (`data/field_photos/`)"),
    code('''
from src.field_test import field_photos
from src.field_test import _load_model, _p_varoa
fotke = field_photos()
pv = {}
for m in MODELI:
    _, model, tf = _load_model(RUN_ZA[m]); pv[m] = _p_varoa(model, tf, [f for f, _ in fotke])
fig, axes = plt.subplots(len(fotke), 2, figsize=(11, 3.6 * len(fotke)), gridspec_kw={"width_ratios": [1.3, 1]})
axes = np.atleast_2d(axes)
for i, (f, t) in enumerate(fotke):
    axes[i, 0].imshow(Image.open(f)); axes[i, 0].axis("off")
    axes[i, 0].set_title(f"prava klasa: {'varoa' if t else 'zdrava'}", color=BOJA_KLASE["varoa" if t else "zdrava"])
    vals = [pv[m][i] for m in MODELI]
    axes[i, 1].barh(MODELI, vals, color=[BOJA_MODELA[m] for m in MODELI]); axes[i, 1].set_xlim(0, 1)
    axes[i, 1].axvline(0.5, color="k", ls="--", lw=1)
    for j, v in enumerate(vals): axes[i, 1].text(min(v + 0.02, 0.85), j, f"{v:.3f}", va="center")
    axes[i, 1].set_title("P(varoa)")
plt.tight_layout(); sacuvaj("05g_terenske_fotografije"); plt.show()
'''),
    md("## 7. Izbor modela — svi kriterijumi"),
    code('''
from src.predictions import predictions as P
lat = pd.read_csv(ROOT / "models" / "latency_benchmark_varroa.csv").set_index("model")
kr = pd.DataFrame({m: {
    "Test F1": json.loads((RUNS / RUN_ZA[m] / "metrics.json").read_text())["macro_f1"],
    "Test tačnost": json.loads((RUNS / RUN_ZA[m] / "metrics.json").read_text())["accuracy"],
    "Test odziv (varoa)": json.loads((RUNS / RUN_ZA[m] / "metrics.json").read_text())["varroa_recall"],
    "Test AUC": json.loads((RUNS / RUN_ZA[m] / "metrics.json").read_text())["roc_auc"],
    "Gleda grinju (1 − ostalo posle brisanja)": 1 - checks[m]["masking"]["ostalo_varoa_kad_se_grinja_prekrije"],
    "BeeDataset AUC": roc_auc_score(bd[m]["true"], bd[m]["p_varoa"]),
    "ONNX ms / slika": lat.loc[m, "onnx_batch1_ms"],
    "Fajl MB": lat.loc[m, "fajl_MB"],
} for m in MODELI}).T
manje_je_bolje = ["ONNX ms / slika", "Fajl MB"]
najbolji = pd.DataFrame({k: (kr[k] == (kr[k].min() if k in manje_je_bolje else kr[k].max())).astype(float) for k in kr})
fig, ax = plt.subplots(figsize=(13, 3.2))
sns.heatmap(najbolji, annot=kr.round(3).astype(str), fmt="", cmap=sns.light_palette("#27ae60", as_cmap=True),
            cbar=False, linewidths=0.8, linecolor="white", annot_kws={"size": 11}, ax=ax)
ax.set_title("Svi kriterijumi izbora (zeleno = najbolji u koloni)"); ax.set_ylabel("")
plt.xticks(rotation=15, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); sacuvaj("05h_svi_kriterijumi"); plt.show()
lat[["onnx_batch1_ms", "fajl_MB"]]
'''),
    md("""
## 8. Odluka: **ResNet50**

- najbolji po **svim metrikama koje je profesorica navela** (tačnost, preciznost, odziv, F1) i po AUC
- najmanje lažnih uzbuna (111, naspram 156 kod MobileNetV3)
- prolazi uzročni test — odluka zavisi od grinje
- tačno prepoznaje terensku fotografiju na kojoj je BeeImage model pogrešio
- sporiji (12 ms) i veći (94 MB), ali za server je to nebitno

**Rezerve:** razlike su ~1 procentni poen (jedno seme); MobileNetV3 se bolje prenosi na BeeDataset i
4,6× je brži — bio bi bolji izbor za mobilni uređaj. Zamena modela u sistemu je zamena jednog fajla.
"""),
    md("## 9. Provera kroz pravi servis (isti kod i ONNX fajl kao u aplikaciji)"),
    code('''
from service.inference import BeeHealthModel
servis = BeeHealthModel(ROOT / "models" / "bee_health.onnx")
print("Model u servisu:", servis.arch, "| klase:", servis.classes, "\\n")
for f, t in fotke:
    print(f"{'varoa ' if t else 'zdrava'}  {Path(f).name:38s} ->", servis.predict(Path(f).read_bytes()))
print("\\nProvera na celom test skupu:", json.loads((ROOT / "models" / "verification.json").read_text()))
'''),
])
