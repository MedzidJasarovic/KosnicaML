# Stanje rada — gde smo stali

Poslednje ažuriranje: **07.10.2026.**

Ovaj dokument služi da se rad nastavi tačno odakle je prekinut.

- **Konačni izveštaj za profesoricu:** [`report/izvestaj_final.pdf`](report/izvestaj_final.pdf)
  i [`report/izvestaj_final.docx`](report/izvestaj_final.docx) (generiše ih `python -m src.make_final_report`)
- Radni dnevnik nalaza: [`report/NALAZI.md`](report/NALAZI.md)
- Notebook-ovi: `notebooks/01_analiza_dataseta.ipynb`, `02_trening_i_poredjenje.ipynb`, `03_provera_modela.ipynb`

---

## Kontekst

Diplomski rad: profesorica je suzila temu na **prepoznavanje zdravstvenog stanja
pčela sa fotografije** (transfer learning, poređenje MobileNetV3/ResNet50/EfficientNetB0,
integracija najboljeg modela u IS „Košnica“). Moduli A (prinos) i C (RAG) iz
`../plan.txt` su **izvan obima**. Konsultacije 06.10.2026: profesorica je saglasna
da se najbolji model integriše i testira na slikama.

## Tok rada (ukratko)

1. **Prvi eksperiment — BeeImage.** 97% tačnosti, ali model je učio pozadinu
   snimka, ne bolest: makro fotografija pčele sa 2 grinje → „zdrava“ 99,97%.
   Dokaz: `python -m src.shortcut_test` (iz samih uglova slike 74% „varoa“).
   Oznake u BeeImage su po snimku, ne po pčeli. NALAZI.md C-X. Modeli arhivirani
   u `models/beeimage/`.
2. **Glavni eksperiment — VarroaDataset** (TU Wien, Zenodo 10.5281/zenodo.4085044):
   13.509 slika, oznaka po pčeli, ručno označene grinje, podela po snimku.
   Klase: `zdrava`, `varoa`. BeeDataset (Hickert) je nezavisan test skup.
3. **Protokol** biran samo na val skupu (`runs/tuning/`, NALAZI C-Y): fine-tuning
   samo poslednjeg bloka ne uči grinju (val F1 0,665); fine-tuning CELE mreže radi
   (0,934). Izabrano: 3 epohe glava + 15 epoha cela mreža, lr 1e-4, letterbox 224.

## Rezultati (test skup, 3.408 slika iz 2 snimka koje model nije video)

| Model | Accuracy | F1 macro | Recall varoa | BeeDataset AUC | Makro foto | ONNX ms |
|---|---|---|---|---|---|---|
| MobileNetV3-Large | 0,920 | 0,901 | 0,876 | **0,949** | ✗ | 2,6 |
| **ResNet50 — izabran** | **0,931** | **0,913** | 0,868 | 0,869 | ✓ | 11,9 |
| EfficientNetB0 | 0,922 | 0,901 | 0,842 | 0,928 | ✓ | 4,5 |

Sva tri modela gledaju grinju: kad se grinja prekrije, „varoa“ ostaje kod 3–4%
pčela; kad se prekrije drugo mesto, kod 96–98% (`src/model_checks.py`).

Slabost: na drugačijem snimanju (BeeDataset) modeli su previše oprezni —
recall varoe 0,20–0,48 na pragu 0,5, iako je AUC visok. Obrazloženje izbora i
rezerve: poglavlje „Izbor modela“ u izveštaju.

## U produkciji

`models/bee_health.onnx` = **ResNet50 (VarroaDataset)**, proveren na celom test
skupu kroz servis (`models/verification.json`: 0 različitih predikcija).
Alternativni modeli: `models/varroa_*.onnx`. Zamena: `python -m src.export_onnx
--run varroa_<arh> --name bee_health` i restart servisa.

Integracija (urađeno 06.10.):
- `service/` — FastAPI (`/health`, `/predict`), priprema slike u `service/preprocessing.py`
  (ista funkcija u treningu i servisu)
- `KosnicaApi` — `BeeHealthChecksController` (`/api/hives/{id}/health-checks`),
  tabela `BeeHealthChecks`, `LocalStorageService` (zamenio Mock; slike u `wwwroot/uploads/`)
- `KosnicaUI` — tab „Zdravlje pčela“ u modalu košnice

## Pokretanje

```bash
# ML servis
cd ~/Kosnica_IS/KosnicaML && ./.venv/bin/uvicorn service.main:app --port 8000
# backend
cd ~/Kosnica_IS/KosnicaApi && dotnet watch
# frontend
cd ~/Kosnica_IS/KosnicaUI && npm start
```

Probne slike iz test skupa: `data/probne_slike/` (prava klasa u imenu fajla).

## Ponavljanje eksperimenta

```bash
./run_varroa.sh                       # ceo eksperiment (~1,5 h na M5); preskače već istrenirane
python -m src.field_test              # posle dodavanja fotografija u data/field_photos/{zdrava,varoa}/
python -m src.make_final_report       # izveštaj (md, docx, pdf)
./.venv/bin/python notebooks/build_notebooks.py && ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Dug trening pokretati sa `nohup ... &` — alat za pozadinske komande u Claude Code
prekida posle 2 sata (desilo se 06.10. usred ResNet50).

## Otvoreno / sledeće

1. **Terenske fotografije** — Medžid planira da skupi fotografije pojedinačnih pčela
   sa različitim pozadinama u `data/field_photos/{zdrava,varoa}/`. Terenski test sada
   ima samo 2 fotografije.
2. Ako terenski test pokaže pad: augmentacija zamenom pozadine ili dodavanje
   BeeDataset-a u trening (Medžid je rekao da je zamena pozadine previše
   komplikovana — predložiti ponovo tek ako rezultati to traže).
3. Snimci ekrana aplikacije za izveštaj → `report/figures/screenshot_*.png`
   (builder ih sam ubacuje u poglavlje o integraciji).
4. Ništa od rada od 06.10. nije commit-ovano (KosnicaML, KosnicaApi, KosnicaUI).
