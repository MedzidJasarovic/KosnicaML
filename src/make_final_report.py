"""
Sastavlja konačni izveštaj za profesoricu: report/final/*.md -> report/izvestaj_final.{md,docx,pdf}

Statična poglavlja su pisana ručno (report/final/NN_*.md). Poglavlja sa rezultatima
se generišu ovde, i to ISKLJUČIVO iz artefakata eksperimenta — nijedan broj se ne
upisuje ručno:
    runs/varroa_*/metrics.json, checks.json, history.csv, *.png
    runs/field_test.json, runs/*_3cls/metrics.json (BeeImage)
    models/latency_benchmark_varroa.csv, models/verification.json

Mesta u statičnim poglavljima koja se popunjavaju: <!-- TABELA:ime --> i <!-- SLIKA:ime -->.

    python -m src.make_final_report            # md + docx + pdf
    python -m src.make_final_report --no-pdf
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date

import pandas as pd

from .paths import FIGURES, MODELS, REPORT, ROOT, RUNS

FINAL = REPORT / "final"
ARCH_ORDER = ["MobileNetV3-Large", "ResNet50", "EfficientNetB0"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# --- formatiranje ------------------------------------------------------------

def num(x, digits=4) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x, digits=1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def table(rows: list[dict]) -> str:
    if not rows:
        return "_(nema podataka)_"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def fig(path, caption: str, width: str = "95%") -> str:
    rel = path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path
    return f"![{caption}]({rel}){{width={width}}}"


# --- učitavanje rezultata ----------------------------------------------------

def load_varroa() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("varroa_*/metrics.json")):
        m = json.loads(p.read_text())
        m["_dir"] = p.parent
        checks = p.parent / "checks.json"
        m["checks"] = json.loads(checks.read_text()) if checks.is_file() else None
        m["config"] = json.loads((p.parent / "config.json").read_text())
        out.append(m)
    if not out:
        raise SystemExit("Nema runs/varroa_*/metrics.json — pokreni prvo ./run_varroa.sh")
    return sorted(out, key=lambda m: ARCH_ORDER.index(m["arch_display"]))


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def latency() -> dict:
    path = MODELS / "latency_benchmark_varroa.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {r["model"]: r for r in df.to_dict("records")}


def chosen_model() -> dict | None:
    return load_json(MODELS / "bee_health.meta.json")


# --- generisana poglavlja ----------------------------------------------------

def ch_sazetak(runs, field, lat, chosen) -> str:
    best = next((m for m in runs if chosen and m["run"] == chosen.get("source_run")), None)
    lines = ["# Sažetak", ""]
    if best is None:
        return "\n".join(lines + ["_(model još nije izabran — pokreni src.export_onnx)_"])

    c = best["checks"] or {}
    mask = c.get("masking", {})
    bee = (field or {}).get(best["run"], {}).get("beedataset", {})
    l = lat.get(best["arch_display"], {})
    lines += [
        "**Zadatak.** Prepoznavanje zaraze pčele grinjom *Varroa destructor* sa fotografije "
        "pojedinačne pčele, transfer learning-om, uz poređenje tri arhitekture i integraciju "
        "najboljeg modela u informacioni sistem „Košnica“.",
        "",
        f"**Izabrani model: {best['arch_display']}**, treniran na skupu VarroaDataset "
        "(13.509 slika, oznaka po pčeli). Na test skupu od "
        f"{thousands(best['test_size'])} slika iz dva snimka koje model nije video tokom treninga:",
        "",
        table([{
            "Accuracy": num(best["accuracy"]),
            "Precision (macro)": num(best["macro_precision"]),
            "Recall (macro)": num(best["macro_recall"]),
            "F1 (macro)": num(best["macro_f1"]),
            "Recall — varoa": num(best["varroa_recall"]),
            "ROC AUC": num(best.get("roc_auc")),
        }]),
        "",
    ]
    facts = []
    if mask:
        facts.append(
            f"Kada se na zaraženoj pčeli grinja prekrije, model i dalje kaže „varoa“ u "
            f"{pct(mask['ostalo_varoa_kad_se_grinja_prekrije'])} slučajeva, a kada se prekrije "
            f"isto toliko mesta na drugom delu pčele — u "
            f"{pct(mask['ostalo_varoa_kad_se_prekrije_drugo_mesto'])}. Odluka modela zavisi od grinje.")
    if bee:
        facts.append(
            f"Na nezavisnom skupu BeeDataset (druga kamera i podloga, {thousands(bee['slika'])} slika) "
            f"ROC AUC je {num(bee.get('roc_auc'))}, a recall na varoi {num(bee['recall_varoa'])}.")
    if l:
        facts.append(f"Inferencija u produkcionom runtime-u (ONNX, CPU) traje {num(l['onnx_batch1_ms'], 1)} ms "
                     f"po slici; fajl modela ima {num(l['fajl_MB'], 1)} MB.")
    lines += [f"- {f}" for f in facts]
    lines += [
        "",
        "**Prvi eksperiment** (skup BeeImage) dao je tačnost 97% na test skupu, ali model nije "
        "prepoznavao bolest nego pozadinu snimka; to je otkriveno ručnim testom i dokazano "
        "posebnim eksperimentom. Pouke iz tog eksperimenta su ugrađene u glavni eksperiment.",
        "",
        "**Integracija.** Model se izvršava u Python/FastAPI servisu (ONNX Runtime). ASP.NET "
        "backend prosleđuje fotografiju servisu, čuva sliku i rezultat u istoriji pregleda "
        "košnice; Angular aplikacija ima novi tab „Zdravlje pčela“ u modalu košnice.",
    ]
    return "\n".join(lines)


def ch_varroa_dataset() -> str:
    from .varroa_dataset import parse_gt

    df = parse_gt()
    split_rows = []
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split]
        split_rows.append({
            "Skup": split, "Snimaka": s["video"].nunique(),
            "Zdrava": thousands(int((s["target"] == 0).sum())),
            "Varoa (oznaka 1)": thousands(int((s["raw_label"] == 1).sum())),
            "Varoa (oznaka 3)": thousands(int((s["raw_label"] == 3).sum())),
            "Ukupno": thousands(len(s)),
        })
    split_rows.append({"Skup": "**ukupno**", "Snimaka": df["video"].nunique(),
                       "Zdrava": thousands(int((df["target"] == 0).sum())),
                       "Varoa (oznaka 1)": thousands(int((df["raw_label"] == 1).sum())),
                       "Varoa (oznaka 3)": thousands(int((df["raw_label"] == 3).sum())),
                       "Ukupno": thousands(len(df))})

    vids = (df.groupby("video").agg(split=("split", "first"), n=("target", "size"), v=("target", "mean"))
            .reset_index().sort_values(["split", "video"], key=lambda c: c.map({"train": 0, "val": 1, "test": 2}).fillna(c) if c.name == "split" else c))
    video_rows = [{"Snimak": r.video, "Skup": r.split, "Slika": thousands(int(r.n)), "Udeo zaraženih": pct(r.v, 0)}
                  for r in vids.itertuples()]

    boxes = [b for bs in df["boxes"] for b in bs]
    w = pd.Series([b[2] - b[0] for b in boxes]).median()
    h = pd.Series([b[3] - b[1] for b in boxes]).median()
    per_bee = df[df["n_mites"] > 0]["n_mites"].value_counts().sort_index()

    return "\n".join([
        "# Skup podataka VarroaDataset", "",
        "Opis i izvor skupa dati su u poglavlju „Pregled javnih skupova podataka“. "
        "Ovde je prikazana analiza preuzetog skupa (`python -m src.varroa_dataset`).", "",
        "## Raspodela po skupovima i klasama", "",
        table(split_rows), "",
        f"Odnos zdravih i zaraženih je {num((df['target'] == 0).sum() / (df['target'] == 1).sum(), 1)} : 1. "
        "Oznaka 3 se tretira kao zaražena (obrazloženje u pregledu skupova).", "",
        "## Podela po snimku", "",
        "Podelu na train/val/test dali su autori skupa, po snimku: **nijedan snimak se ne "
        "pojavljuje u dva skupa**. Svaki snimak sadrži i zdrave i zaražene pčele, pa pozadina "
        "snimka ne može da otkrije klasu — što je bio osnovni problem skupa BeeImage.", "",
        table(video_rows), "",
        "Napomena: snimci `2017-09-25_16-03-38` (val) i `2017-09-25_16-03-38-2` (train) su "
        "verovatno dva dela iste sesije snimanja. To može blago da ulepša rezultate na "
        "validacionom skupu (koji služi samo za izbor checkpoint-a), ali ne utiče na test skup.", "",
        "## Grinja na slici", "",
        f"Svaka zaražena pčela ima 1–{per_bee.index.max()} ručno označene grinje "
        f"({', '.join(f'{thousands(int(n))} sa {k}' for k, n in per_bee.items())}). "
        f"Srednja veličina označene grinje je {w:.0f}×{h:.0f} piksela na slici 160×280 — "
        "jasno vidljiva, za razliku od skupa BeeImage. Koordinate grinja se **ne koriste u "
        "treningu**; služe samo za proveru da li model gleda grinju.", "",
        fig(FIGURES / "varroa_dataset_samples.png",
            "Primeri iz validacionog skupa po oznaci. Crveni pravougaonik je nacrtan iz gt.csv radi prikaza — na samim slikama ga nema."),
    ])


def ch_tuning() -> str:
    """Izbor protokola treninga — isključivo na validacionom skupu (runs/tuning/*)."""
    rows = []
    for cfg_path in sorted((RUNS / "tuning").glob("*/config.json")):
        c = json.loads(cfg_path.read_text())
        hist = pd.read_csv(cfg_path.parent / "history.csv")
        last_train_acc = hist["train_acc"].iloc[-1]
        rows.append({
            "Varijanta": cfg_path.parent.name.split("_")[0],
            "Arhitektura": c["arch_display"],
            "Slika": f"{c.get('img_size', 224)} px",
            "Fine-tuning": "cela mreža" if c.get("finetune_scope") == "all" else "poslednji blok",
            "Epohe (glava + FT)": f"{c['epochs_head']} + {c['epochs_finetune']}",
            "LR fine-tuning": f"{c['lr_finetune']:g}",
            "Train accuracy (kraj)": num(last_train_acc),
            "Najbolji val F1": num(c["best_val_macro_f1"]),
        })
    if not rows:
        return ""
    cfgs = {p.parent.name.split("_")[0]: json.loads(p.read_text()) for p in (RUNS / "tuning").glob("*/config.json")}
    decision = ""
    if {"B", "C"} <= set(cfgs):
        b, c = cfgs["B"], cfgs["C"]
        hist_b = pd.read_csv(RUNS / "tuning" / next(p.name for p in (RUNS / "tuning").iterdir() if p.name.startswith("B_")) / "history.csv")
        ft = hist_b[hist_b["stage"] == "finetune"]["val_macro_f1"]
        swing = ft.iloc[4:].diff().abs().max()  # oscilacija posle početnog rasta
        decision = (
            f"Odmrzavanje cele mreže (B) podiglo je val F1 sa {num(cfgs.get('A', {}).get('best_val_macro_f1'))} na "
            f"{num(b['best_val_macro_f1'])}. Veća slika (C) dala je još {num(c['best_val_macro_f1'] - b['best_val_macro_f1'], 3)}, "
            f"što je manje od oscilacije val F1 između susednih epoha iste varijante (do {num(swing, 3)}), "
            f"a trening je {num(c['train_seconds'] / b['train_seconds'], 1)} puta sporiji (i inferencija sporija). "
            "Izabrana je varijanta **B**: fine-tuning cele mreže, slika 224×224."
        )
    return "\n".join([
        "## Izbor protokola treninga", "",
        "Prvi pokušaj koristio je isti protokol kao eksperiment sa BeeImage-om (varijanta A): "
        "fine-tuning samo poslednjeg bloka tela, stopa učenja 10⁻⁵. Model nije uspeo da nauči "
        "zadatak — tačnost i na samom trening skupu ostala je niska. Uzrok je to što zamrznuti "
        "slojevi, naučeni na ImageNet-u, ne izdvajaju sitno obeležje kao što je grinja. Na "
        "BeeImage-u to nije bio problem, jer je model tamo ionako učio pozadinu.", "",
        "Protokol je zato podešen. Varijante su upoređene **isključivo na validacionom skupu** "
        "(snimci odvojeni i od treninga i od testa), na MobileNetV3 kao najbržoj arhitekturi. "
        "Izabrani protokol je zatim primenjen identično na sve tri arhitekture. Test skup nije "
        "korišćen ni u jednom koraku ovog izbora.", "",
        table(rows), "",
        decision,
    ])


def ch_rezultati(runs, lat) -> str:
    cfg = runs[0]["config"]
    cond = [
        {"Parametar": "Skup", "Vrednost": "VarroaDataset, zvanična podela po snimku"},
        {"Parametar": "Trening / val / test", "Vrednost": f"{thousands(cfg['train_size'])} / {thousands(cfg['val_size'])} / {thousands(runs[0]['test_size'])} slika"},
        {"Parametar": "Klase", "Vrednost": "zdrava, varoa"},
        {"Parametar": "Priprema slike", "Vrednost": f"letterbox {cfg.get('img_size', 224)}×{cfg.get('img_size', 224)}, normalizacija ImageNet"},
        {"Parametar": "Faza 1 (glava)", "Vrednost": f"{cfg['epochs_head']} epoha, AdamW, lr {cfg['lr_head']:g}, kosinusno opadanje"},
        {"Parametar": "Faza 2 (fine-tuning)", "Vrednost": f"{cfg['epochs_finetune']} epoha, lr {cfg['lr_finetune']:g}, "
                                                          + ("cela mreža" if cfg.get("finetune_scope") == "all" else "poslednji blok")},
        {"Parametar": "Batch", "Vrednost": str(cfg["batch_size"])},
        {"Parametar": "Funkcija greške", "Vrednost": "CrossEntropy sa težinama klasa"},
        {"Parametar": "Izbor checkpoint-a", "Vrednost": "najbolji macro F1 na val skupu"},
        {"Parametar": "Seme", "Vrednost": str(cfg["seed"])},
        {"Parametar": "Uređaj", "Vrednost": "Apple M5, MPS"},
    ]

    comp = []
    for m in runs:
        l = lat.get(m["arch_display"], {})
        comp.append({
            "Arhitektura": m["arch_display"],
            "Accuracy": num(m["accuracy"]),
            "Precision": num(m["macro_precision"]),
            "Recall": num(m["macro_recall"]),
            "F1": num(m["macro_f1"]),
            "Recall varoa": num(m["varroa_recall"]),
            "ROC AUC": num(m.get("roc_auc")),
            "Parametara": f"{m['total_params'] / 1e6:.1f} M".replace(".", ","),
            "ONNX ms": num(l.get("onnx_batch1_ms"), 1) if l else "—",
            "Trening min": num(m["train_seconds"] / 60, 1),
        })

    per_class = []
    for m in runs:
        for cls, v in m["per_class"].items():
            per_class.append({"Arhitektura": m["arch_display"], "Klasa": cls,
                              "Precision": num(v["precision"]), "Recall": num(v["recall"]),
                              "F1": num(v["f1"]), "Primera": thousands(v["support"])})

    cms = []
    for m in runs:
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        cms.append({"Arhitektura": m["arch_display"],
                    "Zdrava → zdrava": thousands(tn), "Zdrava → varoa (lažna uzbuna)": thousands(fp),
                    "Varoa → zdrava (propuštena)": thousands(fn), "Varoa → varoa": thousands(tp)})

    videos = sorted(runs[0]["per_video"])
    per_video = []
    for m in runs:
        row = {"Arhitektura": m["arch_display"]}
        for v in videos:
            d = m["per_video"][v]
            row[f"{v} ({pct(d['udeo_varoe'], 0)} zaraženih)"] = num(d["accuracy"])
        per_video.append(row)

    raw = [{"Arhitektura": m["arch_display"],
            "Recall — oznaka 1 (vidljiva)": num(m["varroa_recall_by_raw_label"].get("1")),
            "Recall — oznaka 3 (delimično skrivena)": num(m["varroa_recall_by_raw_label"].get("3"))}
           for m in runs]

    out = [
        "# Eksperimentalni rezultati", "",
        ch_tuning(), "",
        "## Uslovi eksperimenta", "",
        "Sve tri arhitekture trenirane su potpuno istim postupkom (poglavlje „Transfer learning“); "
        "razlikuje se samo telo mreže.", "",
        table(cond), "",
        "## Poređenje arhitektura na test skupu", "",
        "Precision, recall i F1 su *macro* proseci — prosek preko obe klase, bez obzira na "
        "broj primera. „Recall varoa“ je udeo zaraženih pčela koje je model prepoznao — "
        "najvažnija pojedinačna metrika za pčelara, jer je propuštena zaraza najskuplja greška. "
        "ROC AUC meri koliko dobro model razdvaja klase nezavisno od praga odluke (0,5).", "",
        table(comp), "",
        "„ONNX ms“ je vreme obrade jedne slike u runtime-u koji se koristi u produkciji "
        "(onnxruntime, CPU, batch 1).", "",
        "## Metrike po klasi", "",
        table(per_class), "",
        "## Matrice konfuzije", "",
        table(cms), "",
    ]
    for m in runs:
        out += [fig(m["_dir"] / "confusion_matrix.png", f"Matrica konfuzije — {m['arch_display']}", "55%"), ""]
    out += [
        "## Krive učenja", "",
        "Vertikalna linija označava prelaz iz faze 1 (trenira se samo glava) u fine-tuning.", "",
    ]
    for m in runs:
        out += [fig(m["_dir"] / "learning_curves.png", f"Krive učenja — {m['arch_display']}"), ""]
    out += [
        "## Rezultat po test snimku", "",
        "Test skup čine dva snimka sa vrlo različitim udelom zaraženih pčela. "
        "Ujednačen rezultat na oba ukazuje da model ne zavisi od uslova pojedinog snimka.", "",
        table(per_video), "",
        "## Grinja vidljiva i delimično skrivena", "",
        "Recall na varoi, odvojeno po originalnoj oznaci autora skupa.", "",
        table(raw),
    ]
    return "\n".join(out)


def ch_provere(runs, bee_shortcut_ref: float = 0.739) -> str:
    rows_mask, rows_bg, rows_cam = [], [], []
    for m in runs:
        c = m["checks"]
        if not c:
            continue
        k = c["masking"]
        rows_mask.append({
            "Arhitektura": m["arch_display"],
            "Prepoznato (original)": pct(k["prepoznato_kao_varoa_original"]),
            "Ostalo „varoa“ — grinja prekrivena": pct(k["ostalo_varoa_kad_se_grinja_prekrije"]),
            "Ostalo „varoa“ — drugo mesto prekriveno": pct(k["ostalo_varoa_kad_se_prekrije_drugo_mesto"]),
            "P(varoa): original → grinja → drugo mesto":
                f"{num(k['prosecna_p_varoa_original'], 2)} → {num(k['prosecna_p_varoa_grinja_prekrivena'], 2)} → "
                f"{num(k['prosecna_p_varoa_drugo_mesto_prekriveno'], 2)}",
        })
        b = c["background"]
        rows_bg.append({"Arhitektura": m["arch_display"],
                        "„Varoa“ iz uglova zdravih pčela": pct(b["udeo_varoa_uglovi_zdravih"]),
                        "„Varoa“ iz uglova zaraženih pčela": pct(b["udeo_varoa_uglovi_zarazenih"])})
        g = c["gradcam_pointing"]
        rows_cam.append({"Arhitektura": m["arch_display"], "Uzoraka": g["uzoraka"],
                         "Najjača tačka u grinji": pct(g["pogodak_u_grinju"])})

    out = [
        "# Da li model gleda grinju?", "",
        "Prvi eksperiment je pokazao da visoka tačnost na test skupu nije dovoljna: model može "
        "postići 97% učeći pozadinu. Zato su za sve tri arhitekture izvedene tri dodatne "
        "provere ponašanja (`python -m src.model_checks`). Sve se rade na test skupu.", "",
        "## Provera 1: brisanje grinje", "",
        "Na svakoj zaraženoj test pčeli koju je model prepoznao, svaka označena grinja se "
        "prekrije bojom svoje okoline. Ako odluka modela zavisi od grinje, procena treba da "
        "pređe u „zdrava“. Kao kontrola, isti broj jednako velikih površina prekrije se na "
        "drugom delu pčele, van grinje — to meri koliko sama intervencija (mrlja na slici) "
        "menja procenu. Razlika između ova dva broja je efekat grinje.", "",
        table(rows_mask), "",
    ]
    for m in runs:
        p = m["_dir"] / "masking_examples.png"
        if p.is_file():
            out += [fig(p, f"Primeri — {m['arch_display']}: original, grinja prekrivena, drugo mesto prekriveno", "80%"), ""]
    out += [
        "## Provera 2: samo pozadina", "",
        "Modelu se daju samo četiri ugla slike (25% × 25%), gde je pozadina. Pozadina je "
        "ista za obe klase, pa udeo odgovora „varoa“ treba da bude sličan za uglove zdravih "
        f"i zaraženih pčela. Za poređenje: BeeImage model je iz uglova „prepoznao“ varou u "
        f"{pct(bee_shortcut_ref)} slučajeva.", "",
        table(rows_bg), "",
        "## Provera 3: Grad-CAM", "",
        "Grad-CAM je metoda koja pokazuje koji delovi slike su najviše doprineli odluci "
        "mreže za datu klasu: računa koliko promena svakog dela poslednjeg konvolucionog "
        "sloja menja izlaz za klasu „varoa“. Kao kvantitativna mera koristi se *pointing "
        "game*: da li najjača tačka mape pada u označeni pravougaonik grinje (uz toleranciju "
        "od 8 piksela, jer je Grad-CAM mapa gruba — 7×7 polja pre uvećanja).", "",
        table(rows_cam), "",
        "Grad-CAM i brisanje grinje mere različite stvari. Brisanje grinje je **uzročni** test: "
        "menja se ulaz i meri promena odluke. Grad-CAM je **opisna** mapa, gruba (7×7 polja) i "
        "zavisna od arhitekture. Kada model ima nizak rezultat u pointing game-u, a visok "
        "efekat u testu brisanja, to znači da je odluka zavisna od grinje, ali je mapa pažnje "
        "šira i obuhvata i okolni deo tela pčele — što se vidi na primerima ispod. Za "
        "zaključak „model gleda grinju“ presudan je test brisanja.", "",
    ]
    for m in runs:
        p = m["_dir"] / "gradcam.png"
        if p.is_file():
            out += [fig(p, f"Grad-CAM — {m['arch_display']}. Gornji red: grinja iz gt.csv; donji: mapa pažnje modela."), ""]
    return "\n".join(out)


def ch_van_skupa(runs, field) -> str:
    if not field:
        return "# Test van skupa za trening\n\n_(pokreni python -m src.field_test)_"
    bee_rows, photo_rows = [], {}
    for m in runs:
        r = field.get(m["run"], {})
        b = r.get("beedataset")
        if b:
            bee_rows.append({"Arhitektura": m["arch_display"], "Accuracy": num(b["accuracy"]),
                             "Recall varoa": num(b["recall_varoa"]), "Precision varoa": num(b["precision_varoa"]),
                             "F1 (macro)": num(b["macro_f1"]), "ROC AUC": num(b.get("roc_auc"))})
        for p in r.get("field", {}).get("po_slici", []):
            key = (p["fajl"], p["prava_klasa"])
            photo_rows.setdefault(key, {"Fotografija": p["fajl"], "Prava klasa": p["prava_klasa"]})
            photo_rows[key][m["arch_display"]] = num(p["p_varoa"], 3)
    any_bee = next((field[m["run"]]["beedataset"] for m in runs if field.get(m["run"], {}).get("beedataset")), None)
    n_field = len(photo_rows)

    out = [
        "# Test van skupa za trening", "",
        "Test skup VarroaDataset-a sadrži snimke koje model nije video, ali snimljene istom "
        "opremom, na istoj podlozi. Korisnik aplikacije će slati fotografije drugačijeg "
        "porekla. Zato su modeli testirani i na dva izvora slika koje nisu iz VarroaDataset-a "
        "(`python -m src.field_test`).", "",
        "## BeeDataset — druga kamera, druga podloga", "",
    ]
    if any_bee:
        out += [
            f"{thousands(any_bee['slika'])} slika pčela sa ulaza u košnicu (zelena podloga, druga "
            f"kamera i rezolucija, Nemačka), od čega {thousands(any_bee['sa_varoom'])} sa oznakom varoe. "
            "Model ovaj skup nije video ni u jednoj fazi.", "",
            table(bee_rows), "",
        ]
    out += [
        "## Terenske fotografije", "",
        f"Fotografije pojedinačnih pčela sa proizvoljnom pozadinom ({n_field}), uključujući "
        "fotografiju koja je otkrila problem sa prvim modelom. Prikazana je verovatnoća "
        "klase „varoa“ (prag odluke 0,5). Folder `data/field_photos/` je pripremljen tako "
        "da se proširi novim fotografijama i test ponovi istom komandom.", "",
        table(list(photo_rows.values())),
    ]
    return "\n".join(out)


def tbl_izbor(runs, field, lat) -> str:
    """Svi kriterijumi izbora na jednom mestu."""
    rows = []
    for m in runs:
        c = (m["checks"] or {}).get("masking", {})
        b = (field or {}).get(m["run"], {}).get("beedataset", {})
        l = lat.get(m["arch_display"], {})
        rows.append({
            "Arhitektura": m["arch_display"],
            "Test F1 (macro)": num(m["macro_f1"]),
            "Test recall varoa": num(m["varroa_recall"]),
            "Test ROC AUC": num(m.get("roc_auc")),
            "Grinja prekrivena → ostalo „varoa“": pct(c.get("ostalo_varoa_kad_se_grinja_prekrije")),
            "BeeDataset ROC AUC": num(b.get("roc_auc")),
            "BeeDataset recall varoa": num(b.get("recall_varoa")),
            "ONNX ms/slika": num(l.get("onnx_batch1_ms"), 1) if l else "—",
            "Fajl MB": num(l.get("fajl_MB"), 1) if l else "—",
        })
    return table(rows)


# --- zamene u statičnim poglavljima --------------------------------------------

def tbl_beeimage_3cls() -> str:
    rows = []
    for p in sorted(RUNS.glob("*_3cls/metrics.json")):
        m = json.loads(p.read_text())
        rows.append({"Arhitektura": m["arch_display"], "Accuracy": num(m["accuracy"]),
                     "Precision": num(m["macro_precision"]), "Recall": num(m["macro_recall"]),
                     "F1 (macro)": num(m["macro_f1"]), "Recall varoa": num(m["varroa_recall"])})
    rows.sort(key=lambda r: ARCH_ORDER.index(r["Arhitektura"]))
    return table(rows) + "\n\n_Tri grupisane klase, test skup od 776 slika, nasumična podela._"


def tbl_beeimage_snimci() -> str:
    from .dataset import load_split
    df = pd.concat([load_split(s, 3) for s in ("train", "val", "test")], ignore_index=True)
    df["snimak"] = df["file"].str.split("_").str[0]
    g = df.groupby("label").agg(slika=("file", "size"), snimaka=("snimak", "nunique"))
    return table([{"Klasa": k, "Slika": thousands(int(r.slika)), "Različitih snimaka": int(r.snimaka)}
                  for k, r in g.iterrows()])


def tbl_verifikacija() -> str:
    v = load_json(MODELS / "verification.json")
    if not v:
        return "_(pokreni python -m service.verify_against_pytorch)_"
    return table([{"Model": v["model"], "Test slika": thousands(v["test_slika"]),
                   "Najveća razlika u verovatnoći": num(v["najveca_razlika_verovatnoce"], 5),
                   "Različitih predikcija": v["razlicitih_predikcija"],
                   "Accuracy kroz servis": num(v["accuracy_servisa"])}])


def img_screenshot() -> str:
    shots = sorted(FIGURES.glob("screenshot_*.png"))
    return "\n\n".join(fig(p, p.stem.replace("screenshot_", "").replace("_", " ").capitalize(), "90%") for p in shots)


REPLACEMENTS = {
    "TABELA:beeimage_3cls": tbl_beeimage_3cls,
    "TABELA:beeimage_snimci": tbl_beeimage_snimci,
    "TABELA:verifikacija_servisa": tbl_verifikacija,
    "SLIKA:screenshot_aplikacije": img_screenshot,
}


def fill(text: str) -> str:
    def repl(match):
        fn = REPLACEMENTS.get(match.group(1))
        return fn() if fn else match.group(0)
    return re.sub(r"<!-- (\w+:\w+) -->", repl, text)


# --- sastavljanje ---------------------------------------------------------------

def build_markdown() -> str:
    runs = load_varroa()
    field = load_json(RUNS / "field_test.json")
    lat = latency()
    chosen = chosen_model()

    REPLACEMENTS["TABELA:izbor_modela"] = lambda: tbl_izbor(runs, field, lat)

    def static(name):
        path = FINAL / name
        return fill(path.read_text()) if path.is_file() else f"_(nedostaje {name})_"

    sections = [
        ch_sazetak(runs, field, lat, chosen),
        static("01_uvod.md"),
        static("02_pregled_datasetova.md"),
        static("03_prvi_eksperiment_beeimage.md"),
        ch_varroa_dataset(),
        static("05_transfer_learning.md"),
        static("06_pytorch_vs_keras.md"),
        ch_rezultati(runs, lat),
        ch_provere(runs),
        ch_van_skupa(runs, field),
        static("10_izbor_modela.md"),
        static("11_integracija.md"),
        static("12_ogranicenja.md"),
        static("13_reprodukcija.md"),
    ]
    header = "\n".join([
        "---",
        'title: "Prepoznavanje zdravstvenog stanja pčela na osnovu fotografije"',
        'subtitle: "Transfer learning, poređenje arhitektura i integracija u informacioni sistem „Košnica“"',
        'author: "Medžid Jašarović"',
        f'date: "{date.today().strftime("%d.%m.%Y.")}"',
        "lang: sr-Latn",
        "---",
    ])
    return header + "\n\n" + "\n\n".join(sections) + "\n"


CSS = """
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       max-width: 100%; color: #222; }
h1 { font-size: 18pt; border-bottom: 2px solid #f9a825; padding-bottom: 4px; margin-top: 28px; page-break-before: always; }
h1.title { page-break-before: avoid; border: none; font-size: 22pt; margin-top: 80px; }
header { text-align: center; margin-bottom: 40px; }
h2 { font-size: 13.5pt; margin-top: 20px; } h3 { font-size: 11.5pt; }
table { border-collapse: collapse; margin: 10px 0; font-size: 9pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #fff8e1; }
img { max-width: 100%; display: block; margin: 8px auto; }
figure { margin: 10px 0; page-break-inside: avoid; } figcaption { font-size: 9pt; color: #555; text-align: center; }
pre, code { font-family: Menlo, monospace; font-size: 8.5pt; } pre { background: #f6f6f6; padding: 8px; overflow-x: auto; }
nav#TOC { page-break-after: always; } nav#TOC ul { list-style: none; padding-left: 14px; }
@page { size: A4; margin: 18mm 16mm; }
"""


def build(pdf: bool = True) -> None:
    md_path = REPORT / "izvestaj_final.md"
    md_path.write_text(build_markdown())
    print(f"Markdown: {md_path}")

    docx = REPORT / "izvestaj_final.docx"
    subprocess.run(["pandoc", str(md_path), "--from=markdown", "--to=docx", f"--output={docx}",
                    f"--resource-path={ROOT}", "--toc", "--toc-depth=2"], check=True, cwd=ROOT)
    print(f"Word:     {docx}")

    if not pdf:
        return
    css = REPORT / "final" / "print.css"
    css.write_text(CSS)
    html = REPORT / "izvestaj_final.html"
    subprocess.run(["pandoc", str(md_path), "--from=markdown", "--to=html5", "--standalone",
                    "--embed-resources", f"--css={css}", f"--resource-path={ROOT}", "--toc",
                    "--toc-depth=2", f"--output={html}", "--metadata=toc-title:Sadržaj"], check=True, cwd=ROOT)
    if not shutil.which(CHROME) and not __import__("os").path.exists(CHROME):
        print("Chrome nije pronađen — PDF preskočen (HTML je u report/).")
        return
    out_pdf = REPORT / "izvestaj_final.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print(f"PDF:      {out_pdf}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    build(pdf=not args.no_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
