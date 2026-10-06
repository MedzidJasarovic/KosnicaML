# Izbor modela

## Kriterijumi

Model se bira po tri grupe kriterijuma, ovim redom važnosti:

1. **Uspešnost na test skupu** — metrike koje je tražila profesorica: accuracy,
   precision, recall i F1, uz matricu konfuzije. Posebno se gleda recall na
   varoi (propuštena zaraza je najskuplja greška) i broj lažnih uzbuna.
2. **Da li model gleda grinju** i kako se ponaša na slikama drugačijeg porekla
   (BeeDataset, terenske fotografije).
3. **Inženjerski kriterijumi** — brzina inferencije i veličina modela u
   produkcionom okruženju.

## Svi kriterijumi na jednom mestu

<!-- TABELA:izbor_modela -->

## Odluka: ResNet50

Izabran je **ResNet50**.

**Test skup.** ResNet50 je najbolji po svim metrikama koje je profesorica
navela — accuracy, precision, recall i F1 (macro) — kao i po ROC AUC. Ima
najmanje lažnih uzbuna (zdrava pčela proglašena zaraženom): 111, naspram 156 kod
MobileNetV3 i 118 kod EfficientNetB0. Recall na varoi je praktično isti kao kod
MobileNetV3 (razlika je sedam pčela od 942).

**Ponašanje.** Sva tri modela prolaze test brisanja grinje: kada se grinja
prekrije, procena „varoa“ ostaje kod svega 3–4% pčela, a kada se prekrije drugo
mesto na pčeli — kod 96–98%. Odluka sva tri modela zavisi od grinje, a ne od
pozadine. ResNet50 i EfficientNetB0 su ispravno prepoznali terensku fotografiju
pčele sa dve grinje na saću — fotografiju na kojoj je prvi (BeeImage) model
pogrešio — dok MobileNetV3 nije.

**Brzina i veličina.** ResNet50 je najsporiji (oko 12 ms po slici) i najveći
(94 MB). Za ovu primenu to nije ograničenje: korisnik šalje jednu fotografiju i
čeka odgovor preko mreže, pa je razlika između 3 i 12 ms neprimetna, a model
se izvršava na serveru, ne na mobilnom uređaju.

## Rezerve

Odluka nije jednoznačna i treba je čitati uz sledeće:

- **Razlike na test skupu su male** — oko jednog procentnog poena — a svaka
  arhitektura je trenirana jednom. Bez ponavljanja sa više semena ne može se
  tvrditi da je razlika statistički značajna.
- **MobileNetV3 se bolje prenosi na BeeDataset** (ROC AUC 0,95 naspram 0,87 kod
  ResNet50), a 4,6 puta je brži i 5,5 puta manji. Ako bi se model izvršavao na
  mobilnom uređaju, ili ako bi buduće fotografije više ličile na snimke sa ulaza
  u košnicu nego na makro fotografije, MobileNetV3 bi bio bolji izbor.
- **Terenski test ima malo fotografija**; on ilustruje, ali ne dokazuje.

Arhitektura sistema dozvoljava zamenu modela bez izmena u backendu i frontendu
(poglavlje „Integracija“), pa se odluka može promeniti kada terenski test skup
bude veći. Svi modeli su izvezeni: `models/varroa_*.onnx`.

## Poređenje sa prvim eksperimentom

| | BeeImage model | VarroaDataset model (ResNet50) |
|---|---|---|
| Tačnost na svom test skupu | 97% | 93% |
| Test skup | isti snimci kao trening | snimci koje model nije video |
| Iz same pozadine prepoznaje varou | 74% slučajeva | ~1% |
| Grinja prekrivena → i dalje „varoa“ | nije merljivo | 3,9% |
| Terenska fotografija sa dve grinje | „zdrava“ 99,97% | „varoa“ 98,9% |

Novi model ima nižu tačnost na papiru, ali je ona **poštena**: meri prepoznavanje
grinje na novim snimcima, a ne pamćenje snimaka iz treninga.
