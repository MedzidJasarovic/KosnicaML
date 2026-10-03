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
