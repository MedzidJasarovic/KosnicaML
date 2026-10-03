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
