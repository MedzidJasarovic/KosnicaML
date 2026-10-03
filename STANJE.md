# Stanje rada — gde smo stali

Poslednje ažuriranje: **03.10.2026.**

Ovaj dokument služi da se rad nastavi tačno odakle je prekinut.
Detaljni nalazi su u [`report/NALAZI.md`](report/NALAZI.md), gotov izveštaj u
[`report/izvestaj.docx`](report/izvestaj.docx).

---

## Kontekst u jednoj rečenici

Diplomski rad: profesorica je mejlom suzila temu na **klasifikaciju zdravstvenog
stanja pčela sa fotografije** (transfer learning, poređenje tri arhitekture,
integracija najboljeg modela u postojeći IS „Košnica"). Moduli A (predviđanje
prinosa) i C (RAG asistent) iz `../plan.txt` su **izvan obima**.

## Šta je ZAVRŠENO

| Faza | Status | Rezultat |
|---|---|---|
| 0 — Setup | ✅ | venv sa Python 3.12.15, PyTorch 2.14.1, MPS radi |
| 1 — Dataset | ✅ | BeeImage, 5.172 slike, analiziran i podeljen 70/15/15 |
| 2 — Trening | ✅ | 6 eksperimenata (3 arhitekture × 6 i 3 klase), 42,6 min |
| 3 — Evaluacija | ✅ | test skup 776 slika, confusion matrice, metrike po klasi |
| 4 — Izbor + ONNX | ✅ | `models/bee_health.onnx`, 17,1 MB, verifikovan |
| 5 — Izveštaj | ✅ | `report/izvestaj.docx`, 7 poglavlja, 14 grafikona |
| 6 — FastAPI servis | ⬜ | **sledeće** |
| 7 — ASP.NET integracija | ⬜ | |
| 8 — Angular UI | ⬜ | |

**Prekretnica:** faze 6–8 su planirane tek nakon konsultacija kod profesorice,
kad potvrdi izbor klasa i modela.

## Izabrani model

**MobileNetV3-Large, 3 klase** — `models/bee_health.onnx` + `bee_health.meta.json`

| Metrika (test, 776 slika) | Vrednost |
|---|---|
| Accuracy | 0,9704 |
| Macro F1 | 0,9643 |
| Recall na varoi | 0,9747 |
| Latencija (ONNX Runtime, CPU) | 2,52 ms |
| Veličina fajla | 17,1 MB |

Klase, tim redosledom: `zdrava`, `varoa_grinje`, `drugi_problem`.

Izabran jer je prvi po macro F1 i po recall-u na varoi, a 4,7× brži i 5,5× manji
od ResNet50 (koji je po tačnosti bolji za 0,0012 — jedna slika od 776).

## Odluke donete u dogovoru sa Medžidom

1. **PyTorch + torchvision**, ne TensorFlow/Keras. Obrazloženje u
   `report/03_pytorch_vs_keras.md`.
2. **Dva eksperimenta**: svih 6 originalnih klasa i grupisano u 3.
3. **Trening lokalno** na Apple M5 (MPS), ne u Colab-u.
4. **FastAPI + ONNX Runtime** za serviranje, ne ONNX direktno u .NET-u.
5. **Izveštaj u .docx** formatu.
6. **Podela disjunktna po lokaciji je izbačena** — Medžid je odlučio da nam ne
   treba. Kod za nju je uklonjen, degenerisani `loc_*.csv` fajlovi obrisani.
   Nalaz o konfundiranju lokacije i klase ostaje naveden u poglavlju o
   ograničenjima, bez posebnog akcenta.

## Kako nastaviti

```bash
cd ~/Kosnica_IS/KosnicaML
source .venv/bin/activate      # ili koristi ./.venv/bin/python direktno
```

Dataset je na disku (`data/raw/`, 62 MB) i **ne treba ga ponovo preuzimati**.
Kaggle token je obrisan i više nije potreban.

Ponovno pokretanje bilo čega:

```bash
./run_all.sh                                    # svih 6 treninga iznova (~43 min)
./.venv/bin/python -m src.evaluate --all        # evaluacija (~4 min)
./.venv/bin/python -m src.benchmark_onnx        # latencija u ONNX-u
./report/build_docx.sh                          # regeneriši izveštaj
./tests/smoke_test.sh                           # provera celog lanca (sintetički podaci)
```

## Sledeći korak — faza 6

FastAPI servis u `KosnicaML/service/`:
- učita `models/bee_health.onnx` preko `onnxruntime` pri pokretanju
- `POST /predict` — prima sliku, preprocesira po `bee_health.meta.json`,
  vraća `{label, confidence, all_probs}`
- `GET /health`
- Dockerfile

Predložena arhitektura cele integracije je u `report/06_integracija.md`.

## OTVORENO PITANJE za fazu 8

U sistemu „Košnica" **čuvanje fajlova ne radi**:
- `KosnicaApi/Services/MockStorageService.cs` vraća izmišljen Cloudinary URL
  posle `Task.Delay(500)` — nijedan bajt se ne čuva
- `KosnicaApi/Program.cs` ne poziva `app.UseStaticFiles()`
- u `KosnicaUI` ne postoji **nikakav** upload UI

(Napomena: `../implementation_roadmap.md` tvrdi suprotno — da foto prilozi već
postoje u hive-detail modalu. To je netačno.)

**Nedogovoreno:** da li implementaciju skladištenja radimo kao deo faze 8 ili
kao zaseban zadatak pre nje. To je posao nezavisan od modela.
