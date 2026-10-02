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
