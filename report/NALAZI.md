# Nalazi i zapažanja — radni dokument

Hronološki dnevnik zapažanja sa dokazima. Odavde se pune sekcije izveštaja.
Svaki nalaz ima **dokaz** (komanda ili fajl kojim se proverava) da bi se na
konsultacijama mogao potkrepiti, a ne samo tvrditi.

Datum: 03.10.2026. · Hardver: Apple M5, 16 GB RAM · PyTorch 2.14.1, MPS backend

---

## A. Dataset

### A1. Skup je kompletan i konzistentan
5.172 slike, 5.172 reda u `bee_data.csv`, **poklapanje 1:1** — nijedna slika bez
oznake, nijedna oznaka bez slike. Veličina na disku 62 MB.

Licenca **CC0-1.0** (javno dobro) — nema ograničenja za upotrebu u radu.

> Dokaz: `python -c "..."` poređenje `set(csv.file)` i fajlova u `data/raw/bee_imgs/`.
> Napomena: `wc -l` daje 5.171 jer fajl nema prazan red na kraju.

### A2. Kolone u CSV-u
`file, date, time, location, zip code, subspecies, health, pollen_carrying, caste`

Za ovaj rad relevantna je kolona `health`; ostale su iskorišćene za analizu
(`location`, `date`) ili nisu korišćene (`subspecies`, `pollen_carrying`, `caste`).

### A3. Šest klasa i oštar disbalans — odnos 117:1

| `health` vrednost | Slika | Udeo |
|---|---|---|
| `healthy` | 3.384 | 65,4% |
| `few varrao, hive beetles` | 579 | 11,2% |
| `Varroa, Small Hive Beetles` | 472 | 9,1% |
| `ant problems` | 457 | 8,8% |
| `hive being robbed` | 251 | 4,9% |
| `missing queen` | 29 | 0,6% |

Klasa `missing queen` ima samo 29 slika — pri podeli 70/15/15 to je 20 za
trening i **4 za test**. Metrike na toj klasi su statistički bezvredne i to se
mora navesti kao ograničenje.

U datasetu postoji tipfeler: `few varrao` umesto `varroa`. Zbog toga su u
`src/dataset.py` ključevi mapiranja normalizovani (mala slova, sređene praznine),
a ne prepisani iz dokumentacije.

> Dokaz: `report/01_analiza_dataseta.md`, `report/figures/class_distribution.png`

### A4. Slike su varijabilnih dimenzija
Širina 27–346 px (medijana 61), visina 24–340 px (medijana 61). To su **isečci
pojedinačnih pčela**, ne fotografije celog rama.

Posledica: svaka arhitektura skalira slike na svoju ulaznu rezoluciju preko
zvaničnog `weights.transforms()`. Takođe — medijana od 61 px znači da se slike
pri skaliranju na 224×224 **uvećavaju** oko 3,7×, pa se ne dobija nova
informacija, samo interpolacija.

### A5. Oznaka zdravlja je svojstvo SNIMKA, ne pojedinačne pčele
Krstasta tabela lokacija × klasa pokazuje da 5 od 6 klasa dolazi sa tačno
jedne lokacije:

| Lokacija | Klase koje daje |
|---|---|
| Athens, GA, USA | samo `few varroa` (579) |
| Athens, Georgia, USA | samo `Varroa, Small Hive Beetles` (472) |
| Alvin, TX, USA | `ant problems` (457), `hive being robbed` (251), `missing queen` (29) |
| Saratoga / Des Moines / New Orleans / San Jose / Keene | samo `healthy` (ukupno 3.384) |

Dalje: imena fajlova (`041_066.png`) imaju prefiks koji je id snimka — **45
različitih prefiksa, nijedan ne prelazi granicu klase**. Sve 19 kombinacija
(lokacija, datum) sadrži tačno jednu klasu zdravlja. Klasa
`Varroa, Small Hive Beetles` je jedna jedina grupa od 472 skoro identična frejma.

**Posledica za interpretaciju rezultata:** mreža može postići visoku tačnost
učeći pozadinu, osvetljenje i drvenariju pojedinog snimka, a ne obeležja zdravlja
pčele. Zato rezultate treba čitati kao „tačnost na ovom datasetu", a ne kao
„tačnost na novom pčelinjaku".

> Dokaz: `pd.crosstab(df.location, df.label)`

### A6. Podela disjunktna po lokaciji na ovom skupu nije izvodljiva
Pokušana je kao sekundarni eksperiment i degenerisala se:

```
                 train  val  test
ant_problems         0    0   457     <- 457 test primera, 0 za trening
hive_being_robbed    0    0   251     <- isto
missing_queen        0    0    29     <- isto
healthy           2411  973     0
few_varroa         579    0     0
varroa_small       472    0     0
```

Test skup ostaje bez klasa koje u trening skupu imaju nula primera. Eksperiment
je zato izbačen iz obima rada.

---

## B. Metodologija

### B1. Grupisanje 6 → 3 klase
| Grupisana klasa | Originalne klase | Slika |
|---|---|---|
| `zdrava` | `healthy` | 3.384 |
| `varoa_grinje` | `few varrao, hive beetles` + `Varroa, Small Hive Beetles` | 1.051 |
| `drugi_problem` | `ant problems` + `hive being robbed` + `missing queen` | 737 |

Grupisanje smanjuje odnos disbalansa sa 117:1 na 4,6:1.

### B2. Identični uslovi za sve tri arhitekture
Podela (isti fajlovi, `seed=42`), batch 32, 15 epoha glave + 10 epoha
fine-tuninga, AdamW, CosineAnnealingLR, iste augmentacije, izbor checkpointa po
val macro F1. Razlikuje se **isključivo telo mreže**.

> Dokaz: `runs/*/config.json` — svi imaju iste hiperparametre

### B3. Macro F1 kao primarna metrika, ne accuracy
Pri disbalansu 117:1 model koji uvek predviđa `healthy` ima tačnost 65,4% i
nultu upotrebnu vrednost. Macro F1 tretira sve klase jednako. Najbolji
checkpoint se bira po njoj.

Dodatno su uvedene težine klasa u funkciji greške,
`w_i = N / (K · n_i)` — `missing queen` dobija težinu **30,17**, `healthy` 0,25.

---

## C. Rezultati

### C0. TEST rezultati — 776 slika, svaki model evaluiran tačno jednom

Val vrednosti ispod služile su samo za izbor epohe; ovo su merodavne brojke.

**3 klase:**

| Arhitektura | Accuracy | Macro F1 | Recall (varoa) | Macro precision | Macro recall |
|---|---|---|---|---|---|
| MobileNetV3-Large | 0,9704 | **0,9643** | **0,9747** | 0,9529 | 0,9767 |
| ResNet50 | **0,9716** | 0,9625 | 0,9620 | 0,9557 | 0,9698 |
| EfficientNetB0 | 0,9111 | 0,8954 | 0,9241 | 0,8701 | 0,9278 |

**6 klasa:**

| Arhitektura | Accuracy | Macro F1 | Recall (varoa) | Macro precision | Macro recall |
|---|---|---|---|---|---|
| ResNet50 | 0,9072 | **0,8218** | 0,9494 | 0,7796 | 0,9032 |
| MobileNetV3-Large | **0,9124** | 0,8031 | **0,9684** | 0,7753 | 0,8388 |
| EfficientNetB0 | 0,8273 | 0,7020 | 0,9304 | 0,6442 | 0,8200 |

Zapažanja:
- Razlika između MobileNetV3 i ResNet50 na 3 klase je 0,0018 macro F1 — unutar
  šuma jednog pokretanja, pa se ne može tvrditi da je jedan model tačniji.
- **Recall na varoi je visok kod svih modela (0,92–0,97)**, što je dobro jer je
  to najvažnija klasa greške za pčelara.
- EfficientNetB0 je dosledno najslabiji, u obe šeme klasa. Verovatno bi mu
  koristilo podešavanje hiperparametara, ali to bi pokvarilo poređenje pod
  identičnim uslovima.
- Klasa `missing_queen` ima 4 primera u testu: precision 0,40 / recall 0,50 kod
  MobileNetV3, precision 0,44 / recall 1,00 kod ResNet50. Bez statističke težine.
- Prelaz sa 6 na 3 klase donosi +0,16 macro F1 kod MobileNetV3 i +0,14 kod
  ResNet50.

> Dokaz: `runs/summary.csv`, `runs/*/metrics.json`, `runs/*/confusion_matrix.png`

### C0b. Latencija zavisi od RUNTIME-a, ne samo od arhitekture — ključan nalaz

Latencija je merena dva puta, u PyTorch-u i u onnxruntime-u:

| Model | PyTorch (CPU) | ONNX Runtime (CPU) | Ubrzanje |
|---|---|---|---|
| MobileNetV3-Large | 40,9 ms | **2,52 ms** | 16,2× |
| EfficientNetB0 | 85,4 ms | 4,43 ms | 19,3× |
| ResNet50 | 12,7 ms | 11,91 ms | 1,1× |

Arhitekture sa depthwise-separable konvolucijama u PyTorch-u na CPU-u koriste
slabo optimizovane kernele; ONNX Runtime za njih ima znatno bolje. ResNet50
koristi guste konvolucije koje se u oba runtime-a svode na optimizovane GEMM
rutine, pa kod njega razlike skoro nema.

**Zašto je to važno:** po PyTorch merenju ResNet50 izgleda 3× brži od
MobileNetV3; u produkcijskom runtime-u je 4,7× **sporiji**. Merenje latencije u
okviru za trening dalo bi obrnut redosled i pogrešnu odluku o modelu.

> Dokaz: `models/latency_benchmark.csv`, reproducibilno sa `python -m src.benchmark_onnx`

### C1. Rang po ImageNet tačnosti se NE prenosi na ovaj zadatak

| Arhitektura | ImageNet top-1 | Parametara | val macro F1 (3 kl.) | Trening |
|---|---|---|---|---|
| ResNet50 | 80,86% | 25,6M | 0,9673 (test 0,9625) | 12,7 min |
| MobileNetV3-Large | 75,27% | 5,5M | **0,9668** (test **0,9643**) | **3,3 min** |
| EfficientNetB0 | 77,69% | 5,3M | 0,9012 (test 0,8954) | 5,7 min |

MobileNetV3 je praktično sustigao ResNet50 (razlika 0,0005) sa **4,6× manje
parametara** i **3,8× kraćim treningom**, iako je na ImageNet-u slabiji za
5,6 procentnih poena. EfficientNetB0 je, suprotno očekivanju po ImageNet
tačnosti, najslabiji na ovom zadatku.

Ovo je glavni argument za izbor modela koji ide u produkciju: za terensku
upotrebu u informacionom sistemu bitni su i veličina i brzina, a ne samo tačnost.

### C2. Zadatak sa 3 klase je znatno lakši od zadatka sa 6

| Arhitektura | val macro F1 (6 kl.) | val macro F1 (3 kl.) | Razlika |
|---|---|---|---|
| MobileNetV3-Large | 0,8606 | 0,9668 | +0,106 |
| ResNet50 | 0,8128 | 0,9673 | +0,155 |
| EfficientNetB0 | 0,7640 | 0,9012 | +0,137 |

Dobitak je očekivan — grupisanjem se eliminišu klase sa svega 29, 251 i 457
primera, i uklanja razlikovanje između `few varroa` i `Varroa, Small Hive Beetles`
koje je i za čoveka teško.

### C3. Fine-tuning je pomogao u 5 od 6 slučajeva

| Run | Najbolji (samo glava) | Najbolji (+ fine-tuning) | Dobitak |
|---|---|---|---|
| `efficientnet_b0_6cls` | 0,7167 | 0,7640 | **+0,0473** |
| `resnet50_6cls` | 0,7751 | 0,8128 | +0,0377 |
| `resnet50_3cls` | 0,9346 | 0,9673 | +0,0327 |
| `efficientnet_b0_3cls` | 0,8814 | 0,9012 | +0,0198 |
| `mobilenet_v3_large_3cls` | 0,9644 | 0,9668 | +0,0024 |
| `mobilenet_v3_large_6cls` | 0,8606 | 0,8560 | **−0,0046** |

Zapažanje: dobitak od fine-tuninga je najveći kod ResNet50 i EfficientNetB0, a
zanemarljiv ili negativan kod MobileNetV3. Kod `mobilenet_v3_large_6cls` najbolji
rezultat ostaje iz faze sa zamrznutim telom (epoha 8 faze „head") — fine-tuning
ga nije poboljšao. To potkrepljuje da je za MobileNetV3 na ovom skupu dovoljno
koristiti telo kao fiksni ekstraktor obeležja.

> Dokaz: `runs/*/history.csv`, kolone `stage` i `val_macro_f1`

### C4. Ukupno vreme treninga
42,6 minuta za svih 6 eksperimenata na Apple M5 (MPS). Nije bio potreban
eksterni GPU servis.

---

### C-X. Model ne prepoznaje varou na pčeli — uči snimak (06.10.2026)

**Povod:** pri ručnom testu u aplikaciji, makro fotografija pčele sa dve jasno
vidljive varoe dobila je `zdrava` sa 99,97%.

**Test prečice** (`python -m src.shortcut_test`): modelu se daju samo uglovi
test slika (25% × 25%), gde je pozadina snimka, a pčela van kadra.

| Prava klasa | Slika | Cela slika tačno | Samo ugao tačno |
|---|---|---|---|
| zdrava | 507 | 96,4% | 84,0% |
| varoa_grinje | 158 | 97,5% | 73,9% |
| drugi_problem | 111 | 99,1% | 10,8% |

Slučajno pogađanje je 33%. Iz komadića pozadine bez pčele model 74% puta
„pogađa" varou. To potvrđuje A5: oznaka je svojstvo snimka (sve slike varoe su
iz dva snimka u Athens, GA, sa tirkiznom podlogom), a varoa na većini isečaka
nije ni vidljiva. Visoka tačnost na test skupu meri prepoznavanje snimka, ne
bolesti. Model nije upotrebljiv za stvarnu dijagnozu.

Ravne slike jedne boje uvek daju `zdrava`, pa prečica nije samo boja, nego
tekstura i osvetljenje pozadine.

**Posledica:** BeeImage nije pogodan za ovaj zadatak. Potreban je dataset gde je
oznaka varoe dodeljena po pčeli, uz istu pozadinu za sve klase.

### C-Y. VarroaDataset: fine-tuning samo poslednjeg bloka ne uči grinju (06.10.2026)

Protokol preuzet iz BeeImage eksperimenta (glava 15 epoha, pa fine-tuning
poslednjeg bloka sa lr 1e-5) na VarroaDataset-u daje val macro F1 0,665 i
train accuracy svega 0,74 — model ne uči zadatak (underfitting).

Kada se u fazi 2 odmrzne CELA mreža (lr 1e-4), val macro F1 posle 5 epoha je
0,926. Zaključak: zamrznuti ImageNet slojevi ne izdvajaju sitno obeležje
(grinja ~26 px na 224 ulazu). Na BeeImage-u to nije bilo vidljivo jer je model
učio pozadinu, za šta su ImageNet obeležja dovoljna.

Protokol je biran samo na val skupu, na MobileNetV3; isti se primenjuje na sve
tri arhitekture. Varijante su u `runs/tuning/`.

## D. Tehnička zapažanja (za sekciju o implementaciji)

### D1. ONNX izvoz podrazumevano razdvaja težine u zaseban fajl
`torch.onnx.export` sa `external_data=True` (podrazumevano) proizvodi
`model.onnx` od 0,2 MB i `model.onnx.data` od 94 MB. Lokalno radi, pa greška
izgleda kao da je ne postoji — ali kopiranje samo `.onnx` fajla u Docker daje
model bez težina.

Rešenje: `external_data=False` → jedan samostalan fajl. Smoke test aktivno
proverava da `.onnx.data` ne postoji.

### D2. Novi PyTorch ONNX exporter zahteva `onnxscript`
Od verzije 2.9 `torch.onnx.export` podrazumevano koristi dynamo exporter, koji
zavisi od paketa `onnxscript`. Taj paket nije zavisnost PyTorch-a, pa izvoz pada
sa `ModuleNotFoundError` dok se ne instalira eksplicitno.

Takođe: `dynamic_axes` je zastareo kod dynamo exportera. Zamenjen je sa
`dynamic_shapes` u tuple formi — kod dict forme ključevi moraju biti imena
argumenata metode `forward()`, što se razlikuje po arhitekturi.

### D3. Verifikacija ONNX izvoza
Na 20 test slika maksimalna razlika verovatnoća između PyTorch i ONNX modela je
reda **1e-7**, a predikcije se poklapaju 20/20. Tolerancija u skripti je 1e-4.

### D4. Kaggle CLI 2.2 više ne koristi `kaggle.json`
Autentifikacija ide preko `~/.kaggle/access_token`, promenljive
`KAGGLE_API_TOKEN`, ili `kaggle auth login` (OAuth). Stari `kaggle.json` je
nasleđeni format.

### D5. ZIP arhiva raspakuje slike dvostruko ugnežđeno
Dobija se `data/raw/bee_imgs/bee_imgs/`, a CSV referiše samo imena fajlova.
`src/download_data.py` sada sam ravna strukturu.

---

## E. Otvorena pitanja / dalji koraci

- [x] Evaluacija na test skupu (776 slika) — confusion matrice, precision/recall po klasi
- [x] Izbor najboljeg modela: **MobileNetV3-Large, 3 klase** (prvi po macro F1,
      prvi po recall-u na varoi, 4,7× brži i 5,5× manji od ResNet50)
- [x] ONNX izvoz: `models/bee_health.onnx` (17,1 MB, samostalan), verifikovan
- [x] Sve 7 sekcija izveštaja + `report/izvestaj.docx`
- [ ] Posle konsultacija: FastAPI servis, integracija u ASP.NET, Angular UI
- [ ] Opciono: detekcija pčela na slici rama (rešava razliku domena)
