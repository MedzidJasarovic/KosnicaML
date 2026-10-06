# Transfer learning — pristup i primena

## Zašto se mreža ne trenira od nule

Konvolucione neuronske mreže za klasifikaciju slika imaju milione do desetine
miliona parametara i tipično zahtevaju stotine hiljada označenih primera.
Trening skup ovog rada ima 8.225 slika. Treniranje od nule na tolikom skupu
vodi u preprilagođavanje (*overfitting*): mreža zapamti trening slike umesto da
nauči opšte obeležje.

Transfer learning rešava taj problem. Mreže korišćene u ovom radu prethodno su
istrenirane na skupu **ImageNet** (oko 1,28 miliona slika, 1.000 klasa
svakodnevnih objekata). Tokom tog treninga naučile su opšte vizuelne
reprezentacije — niži slojevi prepoznaju ivice, uglove i boje, srednji teksture
i ponavljajuće obrasce, viši delove objekata. Te reprezentacije nisu vezane za
klase ImageNet-a i prenose se na nov domen, u ovom slučaju na fotografije pčela.

Mreža se konceptualno deli na dva dela:

| Deo | Funkcija | Primer: ResNet50 |
|---|---|---|
| Telo (*backbone*) | konvolucioni slojevi koji sliku pretvaraju u vektor obeležja | 23,5 miliona parametara, izlaz 2.048 obeležja |
| Glava (*classifier*) | potpuno povezani sloj koji vektor obeležja preslikava u klase | 2.048 → 1.000 klasa ImageNet-a |

## Tri arhitekture

| Arhitektura | Parametara | Osnovna ideja | Težine (`torchvision`) |
|---|---|---|---|
| **MobileNetV3-Large** | 4,2 M | dubinski razdvojive konvolucije i arhitektura pronađena automatskom pretragom (NAS), projektovana za mobilne uređaje | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| **ResNet50** | 23,5 M | rezidualne veze koje omogućavaju treniranje veoma dubokih mreža; standardna referentna arhitektura | `ResNet50_Weights.IMAGENET1K_V2` |
| **EfficientNetB0** | 4,0 M | uravnoteženo skaliranje dubine, širine i rezolucije | `EfficientNet_B0_Weights.IMAGENET1K_V1` |

## Primenjeni postupak

Postupak je identičan za sve tri arhitekture i sprovodi se u šest koraka.
Implementacija: `src/models.py` i `src/train.py`.

**1. Učitavanje prethodno istreniranog modela** sa ImageNet težinama iz
biblioteke `torchvision`.

**2. Zamena završnog klasifikacionog sloja.** Originalni sloj sa 1.000 izlaza
zamenjuje se novim, slučajno inicijalizovanim, sa dva izlaza — `zdrava` i
`varoa`. Mesto tog sloja razlikuje se po arhitekturi (`model.fc` kod ResNet-a,
`model.classifier[-1]` kod MobileNetV3 i EfficientNetB0), što je u
implementaciji sakriveno iza funkcije `build_model`.

**3. Zamrzavanje prethodno naučenih slojeva.** Svim parametrima tela postavlja
se `requires_grad = False`, pa se trenira samo glava:

| Arhitektura | Ukupno parametara | Trenira se u fazi 1 | Trenira se u fazi 2 |
|---|---|---|---|
| MobileNetV3-Large | 4.204.594 | 1.232.642 (cela glava, dva sloja) | 4.204.594 (sve) |
| ResNet50 | 23.512.130 | 4.098 | 23.512.130 (sve) |
| EfficientNetB0 | 4.010.110 | 2.562 | 4.010.110 (sve) |

**4. Treniranje nove glave** (*feature extraction*): 3 epohe, optimizator
AdamW, stopa učenja 10⁻³ sa kosinusnim opadanjem. Telo u ovoj fazi služi kao
nepromenljiv izvlačilac obeležja. Cilj ove faze nije konačan model, nego da
nova, slučajno inicijalizovana glava dobije razumne težine pre nego što se
odmrzne telo — inače bi njeni veliki, nasumični gradijenti u prvim koracima
fine-tuning-a pokvarili naučene reprezentacije tela.

**5. Fine-tuning cele mreže.** Kad se glava stabilizuje, odmrzava se cela mreža
i trenira još 15 epoha sa stopom učenja 10⁻⁴ — deset puta manjom nego za glavu,
uz kosinusno opadanje. Manja stopa je ključna: veliki koraci u ovoj fazi bi
uništili prethodno naučene reprezentacije (*catastrophic forgetting*).

Odmrzavanje cele mreže, a ne samo poslednjeg bloka, rezultat je eksperimenta
(poglavlje „Eksperimentalni rezultati“, odeljak „Izbor protokola treninga“).
Obeležje koje model treba da prepozna — sitna, ovalna, crvenkasto-braon grinja
od dvadesetak piksela — nije zastupljeno u ImageNet-u, i zamrznuti niži slojevi
ga ne izdvajaju dovoljno dobro: sa odmrznutim samo poslednjim blokom model nije
uspeo da nauči zadatak ni na trening skupu.

**6. Izbor najboljeg checkpoint-a i evaluacija.** Posle svake epohe model se
proverava na validacionom skupu; čuva se stanje sa najboljim macro F1. Test skup
se koristi tačno jednom, na kraju, i ni u jednom trenutku ne utiče na odluke o
modelu.

## Priprema slike (*letterbox*)

Slike u VarroaDataset-u su uspravne, 160×280 piksela. Standardna priprema
slike za ove mreže (`weights.transforms()`) smanjuje kraću stranu na 232
piksela i iseca centralni kvadrat 224×224 — kod uspravne slike to bi odseklo
oko 45% visine, upravo glavu i zadak pčele, gde je grinja često.

Zato se u ovom radu koristi **letterbox**: cela slika se smanji tako da duža
strana bude 224 piksela, a ostatak kvadrata dopuni neutralnom bojom (srednja
boja ImageNet-a, koja posle normalizacije daje vrednosti blizu nule). Ništa se
ne odseca, a isti postupak radi i za fotografije korisnika bilo kog formata.
Normalizacija po kanalu (srednja vrednost i standardna devijacija ImageNet-a)
ostaje ona sa kojom su mreže trenirane.

Funkcija za pripremu slike napisana je **jednom** (`service/preprocessing.py`) i
koriste je i trening i servis u produkciji. Time je isključeno da se slika u
produkciji priprema drugačije nego u treningu — česta i teško uočljiva greška.

## Augmentacija

Na trening skupu svaka slika se pri svakom prolazu nasumično menja, tako da
mreža ne vidi dvaput istu sliku. Augmentacija je identična za sve tri
arhitekture:

| Transformacija | Opseg | Zašto |
|---|---|---|
| preslikavanje horizontalno i vertikalno | p = 0,5 | pčela je snimljena odozgo, pa nema „gore“ |
| rotacija | do ±180° | isto |
| pomeranje i promena razmere | ±5%, 0,8–1,1 | pčela nije uvek u centru i iste veličine |
| osvetljenje, kontrast, zasićenost | ±30% | različita kamera i osvetljenje kod korisnika |
| nijansa (*hue*) | ±2% | malo — crvenkasta boja grinje je signal |
| zamućenje (Gaussian blur) | p = 0,2 | fotografije van fokusa |

![Augmentacija: original (levo) i sedam nasumičnih varijanti iste zaražene pčele](report/figures/varroa_augmentation.png){width=95%}

## Disbalans klasa

U trening skupu je 5.671 zdrava i 2.554 zaražene pčele (2,2 : 1). Model koji
bi uvek govorio „zdrava“ imao bi tačnost 69% bez ikakve upotrebne vrednosti.
Primenjene su dve mere:

1. **Težinska funkcija greške** — `CrossEntropyLoss` sa težinama obrnuto
   proporcionalnim broju primera: `w_i = N / (K · n_i)`. Greška na zaraženoj
   pčeli košta 1,61, na zdravoj 0,73.
2. **Macro F1 kao kriterijum izbora** — checkpoint se bira po macro F1 na
   validacionom skupu, a ne po tačnosti, jer macro F1 jednako tretira obe klase.

## Isti uslovi za sve tri arhitekture

Profesorica je tražila poređenje „pod što sličnijim uslovima“. Za sve tri
arhitekture isti su: podela podataka, priprema slike, augmentacija, veličina
batch-a (32), optimizator, stope učenja, broj epoha u obe faze, težine klasa,
kriterijum izbora checkpoint-a i seme generatora slučajnih brojeva (42).
Razlikuje se jedino telo mreže i, posledično, koji je njegov poslednji blok.
