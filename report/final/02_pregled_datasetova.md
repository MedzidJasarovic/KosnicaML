# Pregled javnih skupova podataka

Za prepoznavanje zdravstvenog stanja pčele sa fotografije potreban je skup slika
u kome je oznaka zdravlja **dodeljena svakoj pčeli posebno, na osnovu onoga što
se na njoj vidi**. Ovaj uslov se pokazao presudnim: prvi korišćeni skup ga ne
ispunjava, i model treniran na njemu nije naučio da prepozna bolest
(poglavlje „Prvi eksperiment: BeeImage“).

## Kriterijumi procene

Svaki skup je procenjen po pet kriterijuma:

1. **Oznaka po pčeli** — da li je oznaka dodeljena pojedinačnoj pčeli na osnovu
   vidljivog obeležja, ili celom snimku/košnici.
2. **Ista pozadina za sve klase** — ako se klase razlikuju po pozadini, mreža
   može naučiti pozadinu umesto bolesti.
3. **Mogućnost poštene podele** — da li postoji podatak o snimku/sesiji, tako da
   test skup sadrži snimke koje model nije video.
4. **Zadatak** — klasifikacija slike (ono što ovaj rad radi) ili detekcija
   objekata (traži drugačiji tip modela).
5. **Dostupnost i licenca.**

## Pronađeni skupovi

| Skup | Slika | Klase | Oznaka po pčeli | Ista pozadina | Podela po snimku | Zadatak |
|---|---|---|---|---|---|---|
| **BeeImage** (Kaggle, Yang 2018) | 5.172 | 6 zdravstvenih stanja košnice | ne — po snimku | ne | ne | klasifikacija |
| **VarroaDataset** (Schurischuster i Kampel, TU Wien 2020) | 13.509 | zdrava / varoa | **da**, sa ručno označenom grinjom | **da** | **da**, zvanična | klasifikacija |
| **BeeDataset** (Hickert, BeeAlarmed 2021) | 7.490 | varoa, polen, hlađenje, osa (više oznaka) | da | da (zelena podloga) | ne | klasifikacija |
| Bilik i sar., *Sensors* 2021 | 600 | zdrava / zaražena pčela, grinja | da | ne | — | detekcija objekata |
| Roboflow Universe (više skupova) | od nekoliko stotina do par hiljada | grinja, pčela | da | ne | — | detekcija objekata |

### BeeImage — *The BeeImage Dataset: Annotated Honey Bee Images* (Kaggle)

Isečci pojedinačnih pčela iz video snimaka na ulazu u košnice na osam lokacija
u SAD. Kolona `health` ima šest vrednosti: *healthy*, *few varroa, hive beetles*,
*Varroa, Small Hive Beetles*, *ant problems*, *hive being robbed*, *missing queen*.

Na prvi pogled odgovara problemu — ima i varou i druga stanja. Detaljna analiza
je pokazala da oznaka opisuje **stanje košnice u trenutku snimanja**, a ne
pojedinačnu pčelu: sve pčele sa istog snimka nose istu oznaku, i na većini
isečaka iz klase „varoa“ grinja se ne vidi. Svaki snimak pripada tačno jednoj
klasi, a klasa varoe potiče iz svega pet snimaka (89% iz dva). Isečci su sitni
(srednja veličina 61×61 piksel), pa je grinja i kad postoji veličine par piksela.

**Procena: nije pogodan** za prepoznavanje bolesti sa fotografije pojedinačne
pčele. Korišćen je u prvom eksperimentu; zašto ne radi, opisano je u posebnom
poglavlju.

### VarroaDataset (TU Wien) — izabran

Schurischuster S., Kampel M. *VarroaDataset*, Zenodo, 2020,
doi: 10.5281/zenodo.4085044, licenca CC BY 4.0.

13.509 slika pojedinačnih pčela rezolucije 160×280 piksela, snimljenih odozgo u
kontrolisanim uslovima, na istoj sivoj podlozi. Pčela je označena kao zaražena
ako se na njoj **vidi** grinja *Varroa destructor*, i svaka vidljiva grinja ima
ručno nacrtan pravougaonik (*bounding box*).

- 9.562 zdrave i 3.947 zaraženih pčela (odnos 2,4 : 1)
- 13 video snimaka iz sedam dana (avgust–oktobar 2017), **svaki snimak sadrži i
  zdrave i zaražene pčele** (od 20% do 100% zaraženih) — pozadina zato ne može
  da otkrije klasu
- zvanična podela na train/val/test od strane autora ide **po snimku**: nijedan
  snimak se ne pojavljuje u dva skupa
- grinja je na slici veličine oko 32 piksela i jasno vidljiva

Dokumentacija opisuje oznake 0 (zdrava) i 1 (zaražena). U fajlu postoji i
nedokumentovana oznaka 3 (864 slike). Zvanična statistika autora (3.947
zaraženih) jednaka je zbiru oznaka 1 i 3, sve slike sa oznakom 3 imaju označenu
grinju, a vizuelno su to slučajevi kada je grinja delimično skrivena (ispod
krila, sa strane). U ovom radu se oznaka 3 tretira kao zaražena, a rezultat se
dodatno prikazuje odvojeno za oznake 1 i 3.

**Procena: pogodan** — ispunjava sva tri ključna uslova. Ograničenje je da ima
samo dve klase (zdrava/varoa) i da su sve slike snimljene u istim uslovima, što
otvara pitanje rada na fotografijama drugačijeg porekla. To se u radu posebno
meri (poglavlje „Test van skupa za trening“).

### BeeDataset (BeeAlarmed)

Hickert F. *Dataset for a camera based bee-hive monitoring*, BeeAlarmed, 2021,
licenca GPL. Dostupan preko TensorFlow Datasets (`bee_dataset`).

7.490 slika pčela na ulazu u košnicu, snimljenih odozgo kroz staklo, na zelenoj
podlozi, sa četiri nezavisne oznake po slici: varoa, polen, hlađenje košnice
krilima, osa. Oznake su dodeljene ručno, po pčeli.

**Procena: pogodan kao nezavisan test skup.** Snimljen je drugom kamerom, u
drugoj zemlji, na drugoj podlozi. Model treniran na VarroaDataset-u, testiran na
BeeDataset-u, daje meru prenosivosti na drugačije snimanje — što je bliže
stvarnoj upotrebi od testa na istom skupu. Nema podatak o snimku, pa nije
pogodan kao primarni skup za trening.

### Skupovi za detekciju objekata

Bilik S. i sar. (*Visual Diagnosis of the Varroa Destructor Parasitic Mite in
Honeybees Using Object Detector Techniques*, Sensors 21(8), 2021) i više skupova
na platformi Roboflow Universe sadrže fotografije sa označenim položajem grinja
i pčela. Namenjeni su modelima za detekciju objekata (YOLO, SSD), koji na slici
pronalaze i lokalizuju svaku grinju. To je drugačiji zadatak od klasifikacije
kojom se bavi ovaj rad. Skupovi na Roboflow-u su različitog kvaliteta, sa
nejasnim poreklom slika i licencama. **Procena: van obima**, ali relevantni za
dalji rad.

## Izbor

Za glavni eksperiment izabran je **VarroaDataset**, a **BeeDataset** se koristi
kao nezavisan test skup. Klasifikacija je time svedena na dve klase — zdrava i
varoa — što je opravdano jer je varoa najznačajniji zdravstveni problem pčela, a
nijedan pronađeni javni skup nema pouzdane oznake po pčeli za druga stanja.
