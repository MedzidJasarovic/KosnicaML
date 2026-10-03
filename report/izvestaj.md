---
title: "Prepoznavanje zdravstvenog stanja pčela na osnovu fotografija"
subtitle: "Transfer learning i poređenje arhitektura — izveštaj o dosadašnjem radu"
author: "Medžid Jašarović"
date: "03.10.2026."
lang: sr
toc: true
---

# Analiza BeeImage dataseta

- Ukupno slika sa oznakom i fajlom na disku: **5172**
- Razlicitih lokacija: **8**
- Razlicitih datuma snimanja: **16**

## Raspodela po 6 originalnih klasa

| originalna `health` vrednost | klasa (6) | broj slika | udeo |
|---|---|---|---|
| healthy | healthy | 3384 | 65.4% |
| few varrao, hive beetles | few_varroa_hive_beetles | 579 | 11.2% |
| varroa, small hive beetles | varroa_small_hive_beetles | 472 | 9.1% |
| ant problems | ant_problems | 457 | 8.8% |
| hive being robbed | hive_being_robbed | 251 | 4.9% |
| missing queen | missing_queen | 29 | 0.6% |

Najbrojnija klasa je healthy (3384 slika), najmanja missing_queen (29 slika) — odnos **117 : 1**.

## Raspodela po 3 grupisane klase

| klasa (3) | originalne klase koje ulaze | broj slika | udeo |
|---|---|---|---|
| zdrava | healthy | 3384 | 65.4% |
| varoa_grinje | few varrao, hive beetles, varroa, small hive beetles | 1051 | 20.3% |
| drugi_problem | ant problems, hive being robbed, missing queen | 737 | 14.2% |

## Dimenzije slika

| dimenzija | min | median | max |
|---|---|---|---|
| sirina | 27 | 61 | 346 |
| visina | 24 | 61 | 340 |

Slike su varijabilnih dimenzija (isecci pojedinacnih pcela), pa ih svaki
model skalira na svoju ulaznu rezoluciju preko zvanicnog `weights.transforms()`.

## Raspodela po lokaciji

| lokacija | broj slika | udeo |
|---|---|---|
| Saratoga, CA, USA | 2000 | 38.7% |
| Des Moines, IA, USA | 973 | 18.8% |
| Alvin, TX, USA | 737 | 14.2% |
| Athens, GA, USA | 579 | 11.2% |
| Athens, Georgia, USA | 472 | 9.1% |
| New Orleans, LA, USA | 170 | 3.3% |
| San Jose, CA, USA | 149 | 2.9% |
| Keene, NH, USA | 92 | 1.8% |

Mali broj lokacija je razlog zasto pored stratifikovane podele pravimo i
podelu disjunktnu po lokaciji — susedni isecci istog snimka inace zavrse
i u train i u test setu, pa je tacnost naduvana.

## Grafikoni

![Raspodela klasa](report/figures/class_distribution.png)

![Primeri po klasi](report/figures/samples_per_class.png)

# Transfer learning — pristup i primena

## Zašto se mreža ne trenira od nule

Konvolucione neuronske mreže za klasifikaciju slika imaju desetine miliona
parametara i tipično zahtevaju stotine hiljada do miliona označenih primera.
Skup kojim raspolažemo ima 5.172 slike, od čega najmanja klasa samo 29 — treniranje
od nule na tolikom skupu neizbežno vodi u preprilagođavanje (overfitting).

Transfer learning rešava upravo taj problem. Mreže korišćene u ovom radu
prethodno su istrenirane na **ImageNet** skupu (oko 1,28 miliona slika,
1.000 klasa svakodnevnih objekata). Tokom tog treninga naučile su opšte
vizualne reprezentacije — niži slojevi detektuju ivice, uglove i boje, srednji
teksture i ponavljajuće obrasce, viši delove objekata. Te reprezentacije nisu
specifične za klase ImageNet-a i prenose se na nov domen, u ovom slučaju na
fotografije pčela.

Mreža se konceptualno deli na dva dela:

| Deo | Funkcija | Primer: ResNet50 |
|---|---|---|
| Telo (*backbone*) | konvolucioni slojevi koji sliku pretvaraju u vektor obeležja | 23,5 miliona parametara, izlaz 2048 obeležja |
| Glava (*classifier*) | potpuno povezan sloj koji vektor obeležja mapira u klase | 2048 → 1.000 ImageNet klasa |

## Primenjeni postupak

Postupak je identičan za sve tri arhitekture i sprovodi se u šest koraka.

**1. Učitavanje prethodno istreniranog modela.** Modeli i njihove ImageNet
težine uzeti su iz `torchvision` biblioteke:

| Arhitektura | Težine |
|---|---|
| MobileNetV3-Large | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| ResNet50 | `ResNet50_Weights.IMAGENET1K_V2` |
| EfficientNetB0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` |

**2. Zamena završnog klasifikacionog sloja.** Originalni sloj sa 1.000 izlaza
zamenjen je novim, slučajno inicijalizovanim, sa brojem izlaza jednakim broju
klasa problema (6 odnosno 3). Mesto tog sloja razlikuje se po arhitekturi —
`model.fc` kod ResNet-a, `model.classifier[-1]` kod MobileNetV3 i EfficientNetB0 —
što je u implementaciji apstrahovano funkcijom `build_model`.

**3. Zamrzavanje prethodno naučenih slojeva.** Svim parametrima tela postavljeno
je `requires_grad = False`, pa je odmrznuta samo nova glava. Kod ResNet50 to
znači da se u ovoj fazi trenira 12.294 od 23.520.326 parametara — oko 0,05%.
Gradijenti se ne propagiraju kroz telo, pa je epoha višestruko brža, a naučene
opšte reprezentacije ostaju nepromenjene.

**4. Trenianje novog klasifikacionog dela** (*feature extraction*). Trenira se
samo glava, 15 epoha, optimizator AdamW sa stopom učenja 1e-3 i kosinusnim
opadanjem. Telo u ovoj fazi služi kao fiksni ekstraktor obeležja.

**5. Fine-tuning dela mreže.** Nakon što se glava stabilizuje, odmrzava se
poslednji konvolucioni blok tela (`layer4` kod ResNet50, poslednja dva bloka
`features` kod MobileNetV3 i EfficientNetB0) i trenira se dodatnih 10 epoha sa
**znatno manjom** stopom učenja (1e-5). Mala stopa je ključna: veliki gradijenti
u ovoj fazi uništili bi naučene reprezentacije (*catastrophic forgetting*).
Niži slojevi ostaju zamrznuti jer su opšte-vizualni i nema koristi od njihovog
prilagođavanja na skupu ove veličine.

Fine-tuning je potreban jer vizualna obeležja relevantna za ovaj problem —
sitna crvenkasto-braon tačka parazita *Varroa destructor* na grudnom košu pčele —
nisu zastupljena u ImageNet domenu, pa ih zamrznuto telo ne izdvaja optimalno.

**6. Evaluacija na izdvojenom test skupu.** Test skup se koristi tačno jednom,
na samom kraju. Validacioni skup služio je isključivo za izbor najboljeg
checkpointa tokom treninga; test skup ni u jednom trenutku nije uticao na
nijednu odluku o modelu.

## Preprocesiranje

Svaka od tri arhitekture trenirana je na ImageNet-u sa svojim parametrima
normalizacije i ulaznom rezolucijom. Ako se na inferenciji primeni pogrešno
preprocesiranje, model radi vidljivo lošije, a greška se ne manifestuje kao
greška u izvršavanju — što je čest i teško uočljiv izvor netačnih rezultata.

U `torchvision`-u svaki objekat težina nosi zvanično preprocesiranje
(`weights.transforms()`), pa je u ovom radu ono uzeto odatle, a ne pisano ručno.
Time je isključena mogućnost da se arhitekture nenamerno porede pod različitim
uslovima.

Augmentacija na trening skupu identična je za sve tri arhitekture:
`RandomResizedCrop` (opseg 0,7–1,0), horizontalno preslikavanje, rotacija do 15°
i varijacija osvetljenja/kontrasta/saturacije do 20%.

## Rukovanje disbalansom klasa

Klasa `healthy` čini 65,4% skupa, a `missing queen` 0,6% — odnos 117:1.
Model koji bi uvek predviđao `healthy` postigao bi tačnost od 65% bez ikakve
upotrebne vrednosti. Zbog toga su primenjene dve mere:

1. **Težinska funkcija greške** — `CrossEntropyLoss` sa težinama obrnuto
   proporcionalnim broju primera po klasi, po formuli
   `w_i = N / (K · n_i)`. Klasa `missing queen` tako dobija težinu 30,17,
   a `healthy` 0,25.
2. **Macro F1 kao primarna metrika** — najbolji checkpoint bira se po macro F1
   na validacionom skupu, ne po tačnosti. Macro F1 tretira sve klase jednako
   bez obzira na njihovu zastupljenost, pa ne nagrađuje ignorisanje malih klasa.

# Izbor okvira: PyTorch ili TensorFlow/Keras

Oba okvira podržavaju sve tri tražene arhitekture sa ImageNet težinama i oba su
u potpunosti dovoljna za ovaj rad. Odabran je **PyTorch sa `torchvision`**
bibliotekom, iz tri razloga specifična za prirodu ovog zadatka.

**1. Preprocesiranje je vezano za težine, ne za programera.** U `torchvision`-u
svaki objekat prethodno istreniranih težina nosi i metodu `transforms()` koja
vraća tačno ono preprocesiranje sa kojim je model treniran. U Keras-u svaka
arhitektura ima zasebnu funkciju `preprocess_input` sa različitim konvencijama —
`EfficientNet` u Keras-u očekuje vrednosti piksela u opsegu 0–255 i normalizaciju
izvodi interno, dok `ResNet50` očekuje već skalirane vrednosti. Zamena tih
funkcija ne izaziva grešku u izvršavanju, samo tiho pogoršava rezultat.
Pošto je cilj ovog rada **poređenje tri arhitekture pod što sličnijim uslovima**,
okvir u kojem je ta klasa greške strukturno onemogućena ima direktnu
metodološku prednost.

**2. Jedna petlja treninga za sve tri arhitekture.** U PyTorch-u se petlja
treninga piše eksplicitno, pa su optimizator, stopa učenja, raspored opadanja,
broj epoha, kriterijum izbora checkpointa i obrada disbalansa doslovno isti kod
za sva tri modela — razlikuje se samo telo mreže. Eksplicitna petlja je ovde
prednost, a ne režijski trošak: čini uslove eksperimenta proverljivim.

**3. Čist put do produkcije.** Model se izvozi u ONNX direktno iz PyTorch-a
(`torch.onnx.export`), bez posrednog koraka. Iz Keras-a je potrebna dodatna
konverzija alatom `tf2onnx`.

Dodatno, PyTorch ima nativnu podršku za **MPS** backend, što omogućava trening
na grafičkom procesoru Apple Silicon računara na kojem je rad izveden, bez
oslanjanja na eksterne GPU servise.

Verzije korišćene u radu: Python 3.12.15, PyTorch 2.14.1, torchvision 0.29.1,
scikit-learn 1.9.1, onnxruntime 1.30.0.

# Eksperimentalni rezultati

## Uslovi eksperimenta

Sve arhitekture trenirane su pod identičnim uslovima; razlikuje se
isključivo telo mreže.

| Parametar | Vrednost |
|---|---|
| Podela podataka | stratifikovano 70/15/15, isti fajlovi za sve modele |
| Ulazna rezolucija | prema `weights.transforms()` svake arhitekture |
| Veličina batch-a | 32 |
| Faza 1 (glava) | 15 epoha, telo zamrznuto, lr 1e-3 |
| Faza 2 (fine-tuning) | 10 epoha, poslednji blok odmrznut, lr 1e-5 |
| Optimizator | AdamW, weight decay 1e-4 |
| Raspored stope učenja | CosineAnnealingLR |
| Funkcija greške | CrossEntropyLoss sa težinama klasa |
| Augmentacija | RandomResizedCrop, flip, rotacija 15°, ColorJitter 0.2 |
| Izbor checkpointa | najveći macro F1 na validacionom skupu |
| Seed | 42 |
| Uređaj | Apple M5, PyTorch MPS |

## Poređenje arhitektura — 6 klasa

Test skup: 776 slika.

| Arhitektura | Accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall | Recall (varoa) | Parametara | Latencija (ms) | Trening (min) |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large | 0.9124 | 0.8031 | 0.9153 | 0.7753 | 0.8388 | 0.9684 | 4.2M | 41.1 | 3.3 |
| ResNet50 | 0.9072 | 0.8218 | 0.9103 | 0.7796 | 0.9032 | 0.9494 | 23.5M | 12.4 | 11.8 |
| EfficientNetB0 | 0.8273 | 0.7020 | 0.8334 | 0.6442 | 0.8200 | 0.9304 | 4.0M | 84.2 | 5.8 |

### MobileNetV3-Large — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.994 | 0.925 | 0.958 | 507 |
| few_varroa_hive_beetles | 0.766 | 0.828 | 0.796 | 87 |
| varroa_small_hive_beetles | 0.753 | 0.859 | 0.803 | 71 |
| ant_problems | 0.793 | 1.000 | 0.885 | 69 |
| hive_being_robbed | 0.946 | 0.921 | 0.933 | 38 |
| missing_queen | 0.400 | 0.500 | 0.444 | 4 |

![Confusion matrix](runs/mobilenet_v3_large_6cls/confusion_matrix.png)

![Krive učenja](runs/mobilenet_v3_large_6cls/learning_curves.png)

### ResNet50 — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.985 | 0.927 | 0.955 | 507 |
| few_varroa_hive_beetles | 0.742 | 0.793 | 0.767 | 87 |
| varroa_small_hive_beetles | 0.750 | 0.845 | 0.795 | 71 |
| ant_problems | 0.840 | 0.986 | 0.907 | 69 |
| hive_being_robbed | 0.917 | 0.868 | 0.892 | 38 |
| missing_queen | 0.444 | 1.000 | 0.615 | 4 |

![Confusion matrix](runs/resnet50_6cls/confusion_matrix.png)

![Krive učenja](runs/resnet50_6cls/learning_curves.png)

### EfficientNetB0 — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.979 | 0.844 | 0.907 | 507 |
| few_varroa_hive_beetles | 0.603 | 0.506 | 0.550 | 87 |
| varroa_small_hive_beetles | 0.573 | 0.887 | 0.696 | 71 |
| ant_problems | 0.731 | 0.986 | 0.840 | 69 |
| hive_being_robbed | 0.679 | 0.947 | 0.791 | 38 |
| missing_queen | 0.300 | 0.750 | 0.429 | 4 |

![Confusion matrix](runs/efficientnet_b0_6cls/confusion_matrix.png)

![Krive učenja](runs/efficientnet_b0_6cls/learning_curves.png)

## Poređenje arhitektura — 3 klasa

Test skup: 776 slika.

| Arhitektura | Accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall | Recall (varoa) | Parametara | Latencija (ms) | Trening (min) |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large | 0.9704 | 0.9643 | 0.9706 | 0.9529 | 0.9767 | 0.9747 | 4.2M | 40.9 | 3.3 |
| ResNet50 | 0.9716 | 0.9625 | 0.9718 | 0.9557 | 0.9698 | 0.9620 | 23.5M | 12.7 | 12.7 |
| EfficientNetB0 | 0.9111 | 0.8954 | 0.9124 | 0.8701 | 0.9278 | 0.9241 | 4.0M | 85.4 | 5.7 |

### MobileNetV3-Large — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.994 | 0.965 | 0.979 | 507 |
| varoa_grinje | 0.917 | 0.975 | 0.945 | 158 |
| drugi_problem | 0.948 | 0.991 | 0.969 | 111 |

![Confusion matrix](runs/mobilenet_v3_large_3cls/confusion_matrix.png)

![Krive učenja](runs/mobilenet_v3_large_3cls/learning_curves.png)

### ResNet50 — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.988 | 0.974 | 0.981 | 507 |
| varoa_grinje | 0.956 | 0.962 | 0.959 | 158 |
| drugi_problem | 0.923 | 0.973 | 0.947 | 111 |

![Confusion matrix](runs/resnet50_3cls/confusion_matrix.png)

![Krive učenja](runs/resnet50_3cls/learning_curves.png)

### EfficientNetB0 — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.970 | 0.895 | 0.931 | 507 |
| varoa_grinje | 0.830 | 0.924 | 0.874 | 158 |
| drugi_problem | 0.811 | 0.964 | 0.881 | 111 |

![Confusion matrix](runs/efficientnet_b0_3cls/confusion_matrix.png)

![Krive učenja](runs/efficientnet_b0_3cls/learning_curves.png)

# Izbor najboljeg modela

## Kriterijumi

Model koji ulazi u informacioni sistem nije nužno onaj sa najvišom tačnošću.
Izbor je izvršen po tri kriterijuma, poređana po važnosti:

1. **Macro F1 na test skupu** — primarna metrika zbog disbalansa klasa.
   Tačnost je neupotrebljiva kao jedini kriterijum: klasa `zdrava` čini 65,4%
   skupa, pa model koji bi uvek predviđao nju imao bi tačnost 65% bez
   ikakve koristi.
2. **Recall na klasi varoe** — lažno negativan nalaz (varoa postoji, model kaže
   da je pčela zdrava) je za pčelara najskuplja greška, jer dovodi do propuštenog
   tretmana i mogućeg gubitka društva. Lažno pozitivan nalaz vodi samo do
   nepotrebne provere.
3. **Latencija i veličina modela** — model se izvršava na serveru pri svakom
   upload-u fotografije i mora se smestiti u Docker sliku.

## Merenje latencije — metodološka napomena

Latencija je merena **dva puta**, u dva različita runtime-a, i rezultati se
bitno razlikuju:

| Model | PyTorch (CPU) | ONNX Runtime (CPU) | Ubrzanje |
|---|---|---|---|
| MobileNetV3-Large | 40,9 ms | **2,52 ms** | 16,2× |
| EfficientNetB0 | 85,4 ms | 4,43 ms | 19,3× |
| ResNet50 | 12,7 ms | 11,91 ms | 1,1× |

Razlog: arhitekture sa *depthwise-separable* konvolucijama (MobileNetV3,
EfficientNetB0) u PyTorch-u na CPU-u koriste slabo optimizovane kernele, dok
ONNX Runtime za njih ima znatno bolje implementacije. ResNet50 koristi klasične
guste konvolucije koje se u oba runtime-a svode na optimizovane GEMM rutine, pa
kod njega razlike skoro nema.

**Posledica:** merenje latencije u okviru za trening dalo bi obrnut redosled i
pogrešnu inženjersku odluku — po PyTorch merenju ResNet50 izgleda 3× brži od
MobileNetV3, dok je u produkcijskom runtime-u 4,7× sporiji. Merodavno je merenje
u runtime-u koji se stvarno izvršava u produkciji.

> Reproducibilno: `python -m src.benchmark_onnx`, rezultat u
> `models/latency_benchmark.csv` (onnxruntime 1.30.0, 100 merenja + 10 warmup)

## Odluka

**Izabran model: MobileNetV3-Large, varijanta sa 3 klase.**

| Kriterijum | MobileNetV3 (3 kl.) | ResNet50 (3 kl.) | EfficientNetB0 (3 kl.) |
|---|---|---|---|
| Macro F1 | **0,9643** | 0,9625 | 0,8954 |
| Recall (varoa) | **0,9747** | 0,9620 | 0,9241 |
| Tačnost | 0,9704 | **0,9716** | 0,9111 |
| Latencija (ONNX) | **2,52 ms** | 11,91 ms | 4,43 ms |
| Veličina modela | **17,1 MB** | 94,2 MB | 16,6 MB |
| Trening | **3,3 min** | 12,7 min | 5,7 min |

MobileNetV3-Large je prvi po dva od tri kriterijuma, i to po dva važnija
(macro F1 i recall na varoi). ResNet50 je bolji samo po tačnosti, i to za
0,0012 — razlika od jedne slike u test skupu od 776, što je unutar šuma.
Istovremeno je MobileNetV3 **4,7× brži** u produkcijskom runtime-u i
**5,5× manji**.

Zašto varijanta sa 3 klase, a ne 6: model sa 6 klasa postiže macro F1 0,8031,
dok onaj sa 3 postiže 0,9643. Razlika dolazi gotovo isključivo od klasa koje
u test skupu imaju 4 (`missing_queen`) i 38 (`hive_being_robbed`) primera —
na njima su metrike statistički nepouzdane. Varijanta sa 3 klase daje pčelaru
upotrebljivu informaciju (zdrava / varoa / drugi problem) sa pouzdanošću koja
se može braniti.

## Izvezeni model

| | |
|---|---|
| Fajl | `models/bee_health.onnx` (17,1 MB, samostalan) |
| Metapodaci | `models/bee_health.meta.json` |
| Klase (u ovom redosledu) | `zdrava`, `varoa_grinje`, `drugi_problem` |
| Ulaz | `input`, NCHW, dinamički batch, 224×224, RGB |
| Izlaz | `logits`, primeniti softmax |
| Preprocesiranje | skaliranje kraće strane na 232, centralni crop 224, deljenje sa 255, normalizacija mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` |

Metapodaci se izvoze zajedno sa modelom namerno: inferencija mora preprocesirati
sliku identično treningu. Odstupanje u preprocesiranju ne izaziva grešku u
izvršavanju, samo tiho pogoršava predikcije — pa je to klasa greške koja se
najteže otkriva.

Verifikacija izvoza: na 20 test slika maksimalna razlika verovatnoća između
PyTorch i ONNX modela je 1,85·10⁻⁵, a predikcije se poklapaju 20/20.

# Integracija modela u informacioni sistem „Košnica"

## Postojeći sistem

| Sloj | Tehnologija |
|---|---|
| Frontend | Angular 17, standalone komponente, TypeScript |
| Backend | ASP.NET Core (.NET 10), REST API, JWT autentifikacija |
| Baza | PostgreSQL, Entity Framework Core |
| Eksterne integracije | OpenWeatherMap |

## Predložena arhitektura

Model se ne izvršava unutar .NET procesa, nego u zasebnom Python servisu.
.NET backend ostaje jedina tačka koju frontend poznaje i jedini sloj koji
pristupa bazi i proverava autorizaciju.

```
Angular (hive-detail modal, tab "Analiza fotografije")
   │  POST /api/predictions/bee-health   (multipart: slika + hiveId, JWT u header-u)
   ▼
ASP.NET Core — PredictionsController
   │  provera vlasništva nad košnicom, snimanje slike, upis rezultata u bazu
   │
   ├─► IStorageService ──────────► trajno skladište slike
   │
   ├─► IBeeHealthService ────────► HTTP POST /predict
   │                                   ▼
   │                            FastAPI servis (Python)
   │                            onnxruntime + bee_health.onnx
   │                            preprocesiranje po bee_health.meta.json
   │                                   │
   │                            { label, confidence, all_probs }
   │                                   ▼
   └─► PostgreSQL: tabela BeeHealthPredictions (istorija pregleda)
```

## Zašto zaseban Python servis, a ne model unutar .NET-a

Razmatrane su dve opcije.

**Opcija A — ONNX Runtime unutar .NET-a** (`Microsoft.ML.OnnxRuntime`).
Prednost: jedan proces, jedan deployment. Nedostatak: preprocesiranje slike
(skaliranje kraće strane, centralni crop, normalizacija po kanalu) moralo bi se
ručno reimplementirati u C#. Svako odstupanje od preprocesiranja iz treninga
tiho pogoršava predikcije, bez greške u izvršavanju.

**Opcija B — Python/FastAPI servis** (izabrano). Preprocesiranje u servisu
koristi iste biblioteke i iste parametre kao trening, pročitane iz
`bee_health.meta.json` koji je nastao pri izvozu modela. Rizik tihog odstupanja
je time eliminisan. Servis je i nezavisno skalabilan, a model se može zameniti
novom verzijom bez ponovnog deploy-a .NET aplikacije.

Model se izvozi u **ONNX**, pa servis koristi `onnxruntime` umesto PyTorch-a —
Docker slika je time reda 100 MB, a ne nekoliko gigabajta. Isti `.onnx` fajl
ostaje upotrebljiv i za Opciju A, ako bi se kasnije prešlo na jedan stek.

## Potrebne promene u backendu

| Komponenta | Opis |
|---|---|
| `Services/IBeeHealthService.cs` + `BeeHealthService.cs` | HTTP klijent ka FastAPI servisu. Po uzoru na postojeći `OpenWeatherMapService`, uključujući njegov obrazac za slučaj nedostupnog servisa. |
| `Models/BeeHealthPrediction.cs` | Nova tabela: `HiveId`, `ImageUrl`, `PredictedLabel`, `Confidence`, `ModelVersion`, `CreatedAt`, `CreatedByUserId`. |
| `Data/ApplicationDbContext.cs` | Novi `DbSet` i EF migracija. Migracije se primenjuju automatski pri pokretanju aplikacije. |
| `Controllers/PredictionsController.cs` | `POST /api/predictions/bee-health` i `GET /api/predictions/hive/{hiveId}` (istorija pregleda po košnici). |

Adresa ML servisa čita se iz promenljive okruženja, po već postojećem obrascu
za konekcioni string baze — ne upisuje se u `appsettings.json`.

Opciono: pri detekciji varoe sa visokom verovatnoćom sistem može automatski
kreirati `Intervention` tipa `Treatment` sa statusom `Planned`, čime predikcija
postaje konkretan zadatak u planeru pčelara.

## Potrebne promene u frontendu

Novi `services/prediction.service.ts` po uzoru na postojeći `hive.service.ts`.
U modalu detalja košnice (`hive-detail-modal`) dodaje se treći tab —
komponenta već koristi obrazac sa `activeTab`. Tab sadrži izbor fajla, prikaz
rezultata kao obojene oznake sa procentom pouzdanosti, i listu prethodnih
analiza za tu košnicu.

## Zavisnost koju treba rešiti pre integracije

Trenutna implementacija skladištenja u sistemu je privremena: servis
`MockStorageService` ne čuva fajl, nego vraća generisan URL, pa u bazu ulazi
samo string. Takođe, u Angular aplikaciji trenutno ne postoji komponenta za
upload fajlova.

Funkcionalnost analize fotografije zato podrazumeva i implementaciju stvarnog
skladištenja (najjednostavnije: lokalni folder posluživan kao statički sadržaj)
i izradu upload komponente. To je deo obima integracije, nezavisan od modela.

# Ograničenja i dalji rad

## Ograničenja skupa podataka

**Oznaka zdravlja je svojstvo snimka, ne pojedinačne pčele.** U BeeImage skupu
5 od 6 klasa dolazi sa tačno jedne lokacije: `few varroa` samo iz Athens (GA),
`Varroa, Small Hive Beetles` samo iz Athens (Georgia), a `ant problems`,
`hive being robbed` i `missing queen` samo iz Alvin (TX). Imena fajlova nose
prefiks snimka — od 45 različitih prefiksa nijedan ne prelazi granicu klase, i
sve 19 kombinacija (lokacija, datum) sadrži tačno jednu klasu.

Zbog toga mreža može deo tačnosti ostvariti prepoznavanjem pozadine, osvetljenja
i opreme pojedinog snimka, a ne obeležja zdravlja pčele. Postignute vrednosti
treba čitati kao tačnost **na ovom skupu podataka**, ne kao očekivanu tačnost na
fotografijama sa novog pčelinjaka.

Kao posledica, podela podataka disjunktna po lokaciji na ovom skupu nije
izvodljiva: svaki pokušaj ostavlja test skup sa klasama koje u trening skupu
imaju nula primera.

**Ekstremni disbalans klasa.** Odnos najbrojnije i najmanje klase je 117:1.
Klasa `missing queen` ima ukupno 29 slika, od čega 4 u test skupu — izmerene
vrednosti precision 0,40 i recall 0,50 na toj klasi nemaju statističku težinu i
navedene su samo radi kompletnosti.

**Rezolucija slika.** Medijana dimenzija je 61×61 piksela, a modeli rade na
224×224. Slike se pri preprocesiranju uvećavaju oko 3,7×, čime se ne dobija nova
informacija. Varroa destructor je na originalnoj slici objekat veličine
nekoliko piksela, što je blizu granice detektabilnosti.

## Ograničenje prenosivosti na terensku upotrebu

Slike u skupu su **isečci pojedinačnih pčela**, kadrirani i centrirani.
Fotografija koju bi pčelar napravio telefonom je snimak celog rama sa desetinama
pčela, drugačijim osvetljenjem i oštrinom. Model treniran na isečcima ne može se
direktno primeniti na takvu fotografiju — to je razlika domena (*domain gap*).

Za terensku upotrebu bio bi potreban prethodni korak detekcije pojedinačnih
pčela na slici (model za detekciju objekata), pa klasifikacija svakog isečka.
To je izvan obima ovog rada, ali je prirodan nastavak.

## Ograničenja eksperimenta

- Hiperparametri (stope učenja, broj epoha, veličina batch-a) nisu optimizovani
  pretragom, nego su postavljeni na standardne vrednosti i **držani identičnim**
  za sve tri arhitekture. To je namerno: cilj je bilo pošteno poređenje
  arhitektura, a ne maksimalna tačnost pojedinog modela. Sa podešavanjem po
  arhitekturi svaki model bi verovatno bio nešto bolji — posebno EfficientNetB0,
  koji je ovde dosledno najslabiji.
- Svaki eksperiment je izvršen jednom, sa fiksiranim seed-om 42. Nije rađeno
  više pokretanja sa različitim seed-ovima, pa razlike manje od oko 0,01 u
  macro F1 ne treba tumačiti kao stvarnu razliku između modela.
- Korišćene su `IMAGENET1K_V2` težine za MobileNetV3 i ResNet50, a `V1` za
  EfficientNetB0 — za taj model druga verzija u `torchvision`-u ne postoji. To je
  mala asimetrija u uslovima koju treba navesti.

## Dalji rad

1. **Integracija u sistem „Košnica"** — FastAPI servis, `PredictionsController`
   u ASP.NET-u, tabela istorije pregleda i tab za analizu fotografije u Angularu.
   Pretpostavlja prethodnu implementaciju stvarnog skladištenja fajlova.
2. **Detekcija pre klasifikacije** — model za detekciju pčela na fotografiji
   rama, da bi se sistem mogao koristiti sa terenskim snimcima.
3. **Nezavisna provera na drugom skupu** — BeeDataset (Hickert, 7.490 slika,
   snimljenih fiksnim rigom na ulazu u košnicu) ima oznaku prisustva varoe i
   mogao bi služiti kao potpuno nezavisan test skup.
4. **Dopuna skupa sopstvenim snimcima** — fotografije sa pčelinjaka korisnika
   sistema, čime bi se smanjila razlika domena i ublažilo konfundiranje
   lokacijom.
