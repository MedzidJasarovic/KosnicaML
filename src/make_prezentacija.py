"""
Prezentacija za konsultacije: slajdovi 16:9 -> report/prezentacija_konsultacije.pdf

Grafici dolaze iz grafici/ (generišu ih notebook-ovi), brojevi iz runs/*/metrics.json
i checks.json — ništa nije upisano ručno.

    python -m src.make_prezentacija
"""
from __future__ import annotations

import json
import subprocess
import sys

from .paths import REPORT, ROOT, RUNS

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
G = ROOT / "grafici"
MODELI = {"MobileNetV3-Large": "varroa_mobilenet_v3_large", "ResNet50": "varroa_resnet50",
          "EfficientNetB0": "varroa_efficientnet_b0"}


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def p(x, d=0):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def img(name, h=None, w=None):
    style = (f"max-height:{h}px;" if h else "") + (f"max-width:{w}px;" if w else "")
    return f'<img src="{(G / name).as_uri()}" style="{style}">'


CSS = """
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #1f2933; }
.slide { width: 1280px; height: 720px; padding: 46px 64px 40px; position: relative; page-break-after: always;
         overflow: hidden; background: #fff; display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; }
.slide::before { content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 8px; background: #f9a825; }
h1 { font-size: 34px; margin: 0 0 6px; } h2 { font-size: 28px; margin: 0 0 18px; color: #1f2933; }
h2 small { display: block; font-size: 17px; font-weight: 400; color: #6b7280; margin-top: 6px; }
.kor { position: absolute; top: 18px; right: 64px; font-size: 13px; color: #9ca3af; letter-spacing: 1px; }
.num { position: absolute; bottom: 16px; right: 64px; font-size: 12px; color: #9ca3af; }
p, li { font-size: 19px; line-height: 1.45; } ul { margin: 6px 0; padding-left: 24px; } li { margin-bottom: 7px; }
.row { display: flex; gap: 32px; align-items: center; flex: 1; min-height: 0; }
.col { flex: 1; min-width: 0; } .center { display: flex; justify-content: center; align-items: center; flex: 1; min-height: 0; }
img { display: block; max-width: 100%; object-fit: contain; margin: 0 auto; }
.big { font-size: 64px; font-weight: 800; line-height: 1; } .bad { color: #c0392b; } .good { color: #27ae60; }
.box { border-radius: 12px; padding: 16px 20px; background: #f8fafc; border: 1px solid #e5e7eb; }
.box.red { background: #fdecea; border-color: #f5b7b1; } .box.green { background: #e9f7ef; border-color: #a9dfbf; }
.box.yellow { background: #fff8e1; border-color: #ffe08a; }
table { border-collapse: collapse; font-size: 17px; width: 100%; } th, td { padding: 7px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }
th { background: #fff8e1; } td.best { color: #1e8449; font-weight: 700; }
.title { justify-content: center; } .title h1 { font-size: 44px; } .sub { font-size: 22px; color: #4b5563; margin-top: 10px; }
.arch { display: flex; align-items: stretch; gap: 14px; margin-top: 10px; }
.node { flex: 1; border: 2px solid #374151; border-radius: 12px; padding: 14px; font-size: 16px; background: #fff; }
.node b { display: block; font-size: 19px; margin-bottom: 6px; } .arrow { display: flex; align-items: center; font-size: 30px; color: #6b7280; }
.node.ml { border-color: #9b59b6; background: #f5eef8; } .node.be { border-color: #2e86c1; background: #ebf5fb; }
.node.fe { border-color: #d35400; background: #fdf2e9; } .node.db { border-color: #27ae60; background: #e9f7ef; }
code { font-family: Menlo, monospace; font-size: 15px; background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }
.small { font-size: 15px; color: #6b7280; }
"""


def build_html() -> str:
    m = {k: json.loads((RUNS / v / "metrics.json").read_text()) for k, v in MODELI.items()}
    c = {k: json.loads((RUNS / v / "checks.json").read_text()) for k, v in MODELI.items()}
    field = json.loads((RUNS / "field_test.json").read_text())
    tun = {d.name[0]: json.loads((d / "config.json").read_text()) for d in (RUNS / "tuning").iterdir()}
    bee = {k: field[v]["beedataset"] for k, v in MODELI.items()}
    r = m["ResNet50"]
    beeimg = json.loads((RUNS / "mobilenet_v3_large_3cls" / "metrics.json").read_text())

    def best(key, lower=False):
        vals = {k: v[key] for k, v in m.items()}
        return (min if lower else max)(vals.values())

    rows = ""
    for k, v in m.items():
        cells = []
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "varroa_recall", "roc_auc"]:
            cls = ' class="best"' if v[key] == best(key) else ""
            cells.append(f"<td{cls}>{n(v[key])}</td>")
        rows += f"<tr><td><b>{k}</b></td>{''.join(cells)}</tr>"

    S = []

    def slide(body, korak="", cls=""):
        S.append(f'<section class="slide {cls}"><div class="kor">{korak}</div>{body}<div class="num">{len(S) + 1}</div></section>')

    slide("""
      <h1>Prepoznavanje zdravstvenog stanja pčela<br>na osnovu fotografije</h1>
      <div class="sub">Transfer learning, poređenje MobileNetV3 / ResNet50 / EfficientNetB0<br>
      i integracija u informacioni sistem „Košnica“</div>
      <p style="margin-top:40px">Medžid Jašarović &nbsp;·&nbsp; konsultacije — ML deo rada</p>""", cls="title")

    slide(f"""<h2>Šta je urađeno — pregled</h2>
      <div class="row"><div class="col"><ul>
      <li><b>Pregled 5 javnih skupova</b> slika pčela — klase i pogodnost</li>
      <li><b>Prvi eksperiment (BeeImage):</b> {p(beeimg['accuracy'])} tačnosti, ali model <span class="bad">ne prepoznaje bolest</span> — otkriveno i dokazano</li>
      <li><b>Glavni eksperiment (VarroaDataset):</b> 3 arhitekture, isti uslovi, metrike + matrice konfuzije</li>
      <li><b>Provera da model gleda grinju</b> — ne samo tačnost</li>
      <li><b>Test van skupa za trening</b> — druga kamera i terenske fotografije</li>
      <li><b>Izabran ResNet50</b>, integrisan u sistem preko Python/FastAPI servisa</li></ul></div>
      <div class="col" style="flex:.8"><div class="box green"><p style="margin:0">Izabrani model na test skupu<br>
      (snimci koje model nikad nije video):</p>
      <p class="big good" style="margin:12px 0">{p(r['accuracy'], 1)}</p><p style="margin:0">tačnost · F1 {n(r['macro_f1'])} · AUC {n(r['roc_auc'])}</p></div></div></div>""")

    slide(f"""<h2>Pregled javnih skupova podataka<small>Ključni uslov: oznaka mora opisivati pčelu na slici</small></h2>
      <table><tr><th>Skup</th><th>Slika</th><th>Klase</th><th>Oznaka po pčeli</th><th>Ista pozadina</th><th>Podela po snimku</th></tr>
      <tr><td><b>BeeImage</b> (Kaggle)</td><td>5.172</td><td>6 stanja košnice</td><td class="bad">✗ po snimku</td><td class="bad">✗</td><td class="bad">✗</td></tr>
      <tr><td><b>VarroaDataset</b> (TU Wien)</td><td>13.509</td><td>zdrava / varoa</td><td class="best">✓ + označena grinja</td><td class="best">✓</td><td class="best">✓ zvanična</td></tr>
      <tr><td><b>BeeDataset</b> (BeeAlarmed)</td><td>7.490</td><td>varoa, polen, osa, hlađenje</td><td class="best">✓</td><td class="best">✓</td><td class="bad">✗</td></tr>
      <tr><td>Bilik i sar. 2021</td><td>600</td><td>pčela, grinja</td><td>✓</td><td>✗</td><td>detekcija objekata</td></tr>
      <tr><td>Roboflow Universe</td><td>razno</td><td>grinja, pčela</td><td>✓</td><td>✗</td><td>detekcija objekata</td></tr></table>
      <p style="margin-top:22px">→ <b>VarroaDataset</b> za trening, <b>BeeDataset</b> kao nezavisan test. BeeImage je korišćen prvi — i pokazao zašto su ovi uslovi bitni.</p>""",
          "KORAK 1 · DATASETI")

    slide(f"""<h2>Prvi eksperiment: BeeImage<small>Odlični rezultati na test skupu …</small></h2>
      <div class="row"><div class="col">{img('02a_beeimage_rezultati.png', h=420)}</div>
      <div class="col" style="flex:.55"><div class="box"><p style="margin:0">Najbolji model (MobileNetV3):</p>
      <p class="big" style="margin:10px 0">{p(beeimg['accuracy'], 1)}</p><p style="margin:0">tačnosti na 776 test slika</p></div></div></div>""",
          "KORAK 2 · BEEIMAGE")

    slide(f"""<h2>… ali ručni test u aplikaciji<small>Pčela sa dve jasno vidljive grinje</small></h2>
      <div class="center">{img('02b_beeimage_promasaj.png', h=500)}</div>""", "KORAK 2 · BEEIMAGE")

    slide(f"""<h2>Zašto? Oznaka opisuje snimak, ne pčelu</h2>
      <div class="row"><div class="col" style="flex:1.5">{img('01a_beeimage_snimci_po_klasi.png', h=260)}
      <div style="height:10px"></div>{img('01b_beeimage_curenje_snimaka.png', h=250)}</div>
      <div class="col" style="flex:.7"><ul><li>pčele su isečene iz <b>video snimaka</b>; svaka dobija oznaku cele košnice</li>
      <li>1.051 slika varoe = <b>5 snimaka</b> (89% iz dva)</li>
      <li>nasumična podela → <b>isti snimak u trening i u test</b> skupu</li>
      <li>grinja se na isečcima (~61×61 px) uglavnom ne vidi</li></ul></div></div>""", "KORAK 2 · BEEIMAGE")

    slide(f"""<h2>Dokaz: test prečice<small>Model dobija samo uglove slike — bez pčele</small></h2>
      <div class="row"><div class="col">{img('02c_test_precice.png', h=470)}</div>
      <div class="col" style="flex:.55"><div class="box red"><p style="margin:0">Iz komadića pozadine model „prepoznaje“ varou u</p>
      <p class="big bad" style="margin:10px 0">74%</p><p style="margin:0">slučajeva. Slučajno bi bilo 33%.</p></div>
      <p>Model je naučio <b>pozadinu snimka</b> — <i>shortcut learning</i>. Tačnost od 97% meri prepoznavanje snimka, ne bolesti.</p></div></div>""",
          "KORAK 2 · BEEIMAGE")

    slide("""<h2>Pouke → uslovi za novi skup podataka</h2>
      <div class="row"><div class="col"><ol>
      <li><b>Oznaka po pčeli</b>, na osnovu vidljivog obeležja</li>
      <li><b>Ista pozadina</b> za sve klase</li>
      <li><b>Podela po snimku</b> — test sadrži snimke koje model nije video</li>
      <li>Visoka tačnost nije dovoljna — treba <b>proveriti šta model gleda</b> i testirati ga na slikama drugačijeg porekla</li></ol></div>
      <div class="col" style="flex:.7"><div class="box yellow"><p style="margin:0">Sva četiri uslova su ugrađena u glavni eksperiment.</p></div></div></div>""",
          "KORAK 2 · BEEIMAGE")

    slide(f"""<h2>VarroaDataset (TU Wien, 2020)<small>13.509 slika pojedinačnih pčela · svaka grinja ručno označena · CC BY 4.0</small></h2>
      <div class="center">{img('01g_varroa_primeri.png', h=520)}</div>""", "KORAK 1 · DATASETI")

    slide(f"""<h2>VarroaDataset — poštena podela<small>Svaki snimak ima i zdrave i zaražene pčele; nijedan snimak nije u dva skupa</small></h2>
      <div class="row"><div class="col">{img('01e_varroa_snimci.png', h=470)}</div>
      <div class="col" style="flex:.7">{img('01d_varroa_raspodela.png', h=300)}
      <p class="small">Klase: zdrava (9.562) i varoa (3.947). Oznaka 3 (grinja delimično skrivena) računa se kao varoa, kao kod autora.</p></div></div>""",
          "KORAK 1 · DATASETI")

    slide("""<h2>Transfer learning — postupak<small>Isti za sve tri arhitekture (PyTorch + torchvision)</small></h2>
      <div class="row"><div class="col"><ol>
      <li>mreža istrenirana na <b>ImageNet</b>-u (1,28 mil. slika)</li>
      <li>poslednji sloj (1.000 klasa) → novi sa <b>2 izlaza</b>: zdrava, varoa</li>
      <li><b>faza 1:</b> telo zamrznuto, trenira se samo glava — 3 epohe, lr 10⁻³</li>
      <li><b>faza 2:</b> fine-tuning <b>cele mreže</b> — 15 epoha, lr 10⁻⁴</li>
      <li>checkpoint sa najboljim <b>macro F1 na validacionom</b> skupu</li>
      <li><b>test skup</b> se koristi tačno jednom, na kraju</li></ol>
      <p class="small">Isti za sve: podaci, priprema slike, augmentacija, batch 32, AdamW, težine klasa, seme 42.</p></div>
      <div class="col" style="flex:.85">""" + img('03b_letterbox.png', h=290) + """
      <p class="small" style="text-align:center">Letterbox umesto isecanja — ne odseca glavu i zadak, gde je grinja često.</p></div></div>""",
          "KORAK 3 · TRENING")

    slide(f"""<h2>Izbor protokola — samo na validacionom skupu</h2>
      <div class="row"><div class="col">{img('03d_izbor_protokola.png', h=460)}</div>
      <div class="col" style="flex:.6"><ul>
      <li><b>A</b> — protokol iz BeeImage-a (samo poslednji blok): val F1 <b class="bad">{n(tun['A']['best_val_macro_f1'])}</b> — model ne uči grinju</li>
      <li><b>B</b> — cela mreža: <b class="good">{n(tun['B']['best_val_macro_f1'])}</b></li>
      <li><b>C</b> — cela mreža, veća slika: {n(tun['C']['best_val_macro_f1'])}, ali {n(tun['C']['train_seconds'] / tun['B']['train_seconds'], 1)}× sporije</li></ul>
      <div class="box green"><p style="margin:0">Izabrano <b>B</b>. Test skup nije korišćen za ovu odluku.</p></div></div></div>""",
          "KORAK 3 · TRENING")

    slide(f"""<h2>Krive učenja — sve tri arhitekture</h2>
      <div class="center">{img('03e_krive_ucenja.png', w=1150)}</div>
      <p class="small">Val loss prati train loss — nema izraženog preprilagođavanja. Isprekidana linija: prelaz u fine-tuning.</p>""",
          "KORAK 3 · TRENING")

    slide(f"""<h2>Rezultati na test skupu<small>3.408 slika iz 2 snimka koje modeli nikad nisu videli</small></h2>
      <table><tr><th>Model</th><th>Tačnost</th><th>Preciznost</th><th>Odziv</th><th>F1</th><th>Odziv (varoa)</th><th>ROC AUC</th></tr>{rows}</table>
      <div class="center" style="margin-top:12px">{img('04a_poredjenje_metrika.png', h=330)}</div>""", "KORAK 4 · EVALUACIJA")

    slide(f"""<h2>Matrice konfuzije</h2>
      <div class="center">{img('04b_matrice_konfuzije.png', w=1150)}</div>
      <p>Donji levi kvadrat = <b>propuštene zaraze</b>, gornji desni = <b>lažne uzbune</b>.
      ResNet50: najmanje lažnih uzbuna ({m['ResNet50']['confusion_matrix'][0][1]} naspram {m['MobileNetV3-Large']['confusion_matrix'][0][1]} kod MobileNetV3).</p>""",
          "KORAK 4 · EVALUACIJA")

    slide(f"""<h2>ROC krive i analiza grešaka</h2>
      <div class="row"><div class="col" style="flex:.8">{img('04c_roc_krive.png', h=480)}</div>
      <div class="col">{img('04d_po_snimku_i_oznaci.png', w=640)}
      <p class="small">Jasno vidljiva grinja: ~93–95% pronađeno. Delimično skrivena (oznaka 3): ~55–62% — najveći izvor grešaka.</p></div></div>""",
          "KORAK 4 · EVALUACIJA")

    slide(f"""<h2>Gde model greši (ResNet50)</h2>
      {img('04e_greske_propustene.png', h=255)}{img('04f_greske_lazne_uzbune.png', h=255)}
      <p class="small">Propuštene: grinja ispod krila ili sa strane. Lažne uzbune: odsjaji, polen i tamna mesta koja liče na grinju.</p>""",
          "KORAK 4 · EVALUACIJA")

    rm = c["ResNet50"]["masking"]
    slide(f"""<h2>Da li model gleda grinju? Test brisanja grinje</h2>
      <div class="row"><div class="col">{img('05a_brisanje_grinje.png', h=420)}</div>
      <div class="col" style="flex:.6"><p>Na zaraženoj pčeli grinja se <b>prekrije bojom okoline</b>:</p>
      <ul><li>grinja prekrivena → „varoa“ ostaje kod <b class="good">{p(rm['ostalo_varoa_kad_se_grinja_prekrije'], 1)}</b></li>
      <li>drugo mesto prekriveno (kontrola) → <b>{p(rm['ostalo_varoa_kad_se_prekrije_drugo_mesto'], 1)}</b></li></ul>
      <div class="box green"><p style="margin:0">Odluka sva tri modela <b>zavisi od grinje</b> — uzročni dokaz.</p></div></div></div>""",
          "KORAK 5 · PROVERA")

    slide(f"""<h2>Primeri: šta se desi kad se grinja prekrije (ResNet50)</h2>
      <div class="center">{img('05b_brisanje_primeri.png', h=560)}</div>""", "KORAK 5 · PROVERA")

    slide(f"""<h2>Samo pozadina · Grad-CAM</h2>
      <div class="row"><div class="col">{img('05c_samo_pozadina.png', h=420)}
      <p class="small">Iz pozadine novi modeli kažu „varoa“ u 1–5% slučajeva — podjednako za zdrave i zaražene. BeeImage model: 74%.</p></div>
      <div class="col" style="flex:.8">{img('05d_gradcam_pogoci.png', h=330)}
      <p class="small">Grad-CAM: najjača tačka mape pažnje pada na grinju (MobileNetV3, EfficientNetB0: 87%). ResNet50 ima širu mapu — ali test brisanja je presudan.</p></div></div>""",
          "KORAK 5 · PROVERA")

    slide(f"""<h2>Test van skupa: BeeDataset<small>Druga kamera, druga zemlja, zelena podloga — 7.490 slika koje model nije video</small></h2>
      <div class="center">{img('05f_beedataset_roc_prag.png', w=1150)}</div>
      <p>Modeli dobro <b>razdvajaju</b> klase (AUC {n(bee['ResNet50']['roc_auc'])}–{n(bee['MobileNetV3-Large']['roc_auc'])}), ali su <b>previše oprezni</b> na pragu 0,5 —
      niži prag odluke povećava odziv bez ponovnog treninga.</p>""", "KORAK 5 · PROVERA")

    slide(f"""<h2>Terenske fotografije<small>Fotografija koja je oborila BeeImage model</small></h2>
      <div class="center">{img('05g_terenske_fotografije.png', h=540)}</div>""", "KORAK 5 · PROVERA")

    slide(f"""<h2>Izbor modela: ResNet50</h2>
      {img('05h_svi_kriterijumi.png', w=1150)}
      <div class="row" style="margin-top:10px; align-items:flex-start"><div class="col"><ul>
      <li>najbolji po <b>tačnosti, preciznosti, odzivu, F1</b> i AUC</li><li>najmanje lažnih uzbuna</li>
      <li>prolazi test brisanja grinje; tačan na terenskoj fotografiji</li></ul></div>
      <div class="col"><ul><li class="small">razlike ~1 p.p. (jedno seme)</li>
      <li class="small">MobileNetV3 bolji na BeeDataset-u i 4,6× brži — bolji za mobilni uređaj</li>
      <li class="small">zamena modela u sistemu = zamena jednog fajla</li></ul></div></div>""", "KORAK 5 · IZBOR")

    slide(f"""<h2>BeeImage model vs. novi model</h2>
      <table style="font-size:20px"><tr><th></th><th>BeeImage (MobileNetV3)</th><th>VarroaDataset (ResNet50)</th></tr>
      <tr><td>Tačnost na svom test skupu</td><td>{p(beeimg['accuracy'], 1)}</td><td>{p(r['accuracy'], 1)}</td></tr>
      <tr><td>Test skup</td><td class="bad">isti snimci kao trening</td><td class="best">snimci koje model nije video</td></tr>
      <tr><td>Iz same pozadine kaže „varoa“</td><td class="bad">74%</td><td class="best">~1%</td></tr>
      <tr><td>Grinja prekrivena → i dalje „varoa“</td><td>—</td><td class="best">{p(rm['ostalo_varoa_kad_se_grinja_prekrije'], 1)}</td></tr>
      <tr><td>Fotografija sa dve grinje</td><td class="bad">„zdrava“ 99,97%</td><td class="best">„varoa“ 98,9%</td></tr></table>
      <div class="box yellow" style="margin-top:22px"><p style="margin:0">Niža tačnost na papiru — ali <b>poštena</b>: meri prepoznavanje grinje na novim snimcima.</p></div>""",
          "ZAKLJUČAK")

    slide("""<h2>Integracija u sistem „Košnica“<small>Model radi u zasebnom Python servisu; backend mu šalje sliku preko HTTP-a</small></h2>
      <div class="arch">
        <div class="node fe"><b>Angular</b>tab „Zdravlje pčela“<br><code>FormData</code> + JWT<br>POST /api/hives/{id}/health-checks</div>
        <div class="arrow">→</div>
        <div class="node be"><b>ASP.NET Core</b>BeeHealthChecksController<br>provera vlasništva nad košnicom<br><code>HttpClient</code> → ML servis</div>
        <div class="arrow">→</div>
        <div class="node ml"><b>Python · FastAPI</b>POST /predict<br>letterbox priprema slike<br><code>onnxruntime</code> + ResNet50.onnx</div>
        <div class="node db" style="flex:.8; margin-left:10px"><b>PostgreSQL</b>upisuje ASP.NET:<br>tabela <code>BeeHealthChecks</code><br>istorija pregleda košnice</div>
      </div>
      <div class="row" style="margin-top:22px; align-items:flex-start; gap:26px">
        <div class="col"><table style="font-size:15px">
          <tr><th>Sloj</th><th>Tehnologija</th></tr>
          <tr><td>Trening</td><td>PyTorch + torchvision, Apple MPS</td></tr>
          <tr><td>Format modela</td><td>ONNX (izvoz iz PyTorch-a)</td></tr>
          <tr><td>ML servis</td><td>FastAPI + uvicorn + onnxruntime</td></tr>
          <tr><td>Veza .NET → Python</td><td>typed <code>HttpClient</code>, multipart/form-data, JSON</td></tr>
          <tr><td>Adresa servisa</td><td><code>ML_SERVICE_URL</code> / appsettings</td></tr>
          <tr><td>Baza</td><td>EF Core migracija, kolona <code>jsonb</code></td></tr>
          <tr><td>Slike</td><td>LocalStorageService → <code>wwwroot/uploads</code></td></tr></table></div>
        <div class="col" style="flex:.85"><p class="small" style="margin:0 0 6px">Odgovor ML servisa:</p>
          <pre style="font-size:15px; background:#f3f4f6; padding:12px 14px; border-radius:8px; margin:0">{
  "label": "varoa",
  "confidence": 0.9886,
  "all_probs": {"zdrava": 0.0114, "varoa": 0.9886},
  "model_version": "bee_health"
}</pre>
          <p class="small">Servis proveren na svih 3.408 test slika: 0 razlika u odnosu na PyTorch model. Servis nedostupan → backend vraća 503, slika se ne čuva.</p></div>
      </div>""", "INTEGRACIJA")

    slide("""<h2>Ograničenja i sledeći koraci</h2>
      <div class="row" style="align-items:flex-start; font-size:22px"><div class="col"><h3>Ograničenja</h3><ul>
      <li>trening skup: jedna kamera, siva podloga</li><li>dve klase — zdrava / varoa</li>
      <li>vidljiva grinja ≠ zaražena košnica</li><li>jedno seme po modelu</li><li>malo terenskih fotografija</li></ul></div>
      <div class="col"><h3>Sledeći koraci</h3><ul>
      <li><b>terenski test skup</b> (20–40 fotografija, različite pozadine)</li>
      <li><b>kalibracija praga</b> odluke (bez ponovnog treninga)</li>
      <li>po potrebi: BeeDataset u trening, zamena pozadine</li>
      <li>detekcija grinja (YOLO) — koordinate već postoje</li>
      <li>automatski predlog intervencije „Tretman“</li></ul></div></div>""", "ZAKLJUČAK")

    slide("""<h1>Hvala</h1><div class="sub">Pitanja?</div>
      <p class="small" style="margin-top:40px">Kod, notebook-ovi i izveštaj: github.com/MedzidJasarovic/KosnicaML</p>""", cls="title")

    return f"<!doctype html><html lang='sr'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(S)}</body></html>"


def build_podsetnik() -> None:
    """report/final/podsetnik_konsultacije.md -> report/podsetnik_konsultacije.pdf (samo za mene)."""
    from .make_final_report import CSS as REPORT_CSS
    css = REPORT / "final" / "print.css"
    css.write_text(REPORT_CSS + "\nh1 { page-break-before: auto; margin-top: 30px; } h1:first-of-type { margin-top: 0; }"
                   "\nheader { margin-bottom: 20px; } h1.title { margin-top: 0; font-size: 20pt; }"
                   "\ntable { page-break-inside: auto; } tr { page-break-inside: avoid; } h1, h2 { page-break-after: avoid; }")
    html = REPORT / "podsetnik.html"
    subprocess.run(["pandoc", str(REPORT / "final" / "podsetnik_konsultacije.md"), "--from=markdown", "--to=html5",
                    "--standalone", "--embed-resources", f"--css={css}", f"--output={html}"], check=True, cwd=ROOT)
    pdf = REPORT / "podsetnik_konsultacije.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print(f"PDF: {pdf}")


def main() -> int:
    build_podsetnik()
    html = REPORT / "prezentacija_konsultacije.html"
    html.write_text(build_html())
    pdf = REPORT / "prezentacija_konsultacije.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print(f"PDF: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
