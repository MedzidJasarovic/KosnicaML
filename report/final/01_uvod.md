# Uvod

## Cilj rada

Cilj rada je da se u postojeći informacioni sistem „Košnica“ — veb aplikaciju za
vođenje pčelinjaka, košnica, intervencija i prinosa — doda funkcionalnost
**procene zdravstvenog stanja pčele na osnovu fotografije**. Korisnik učita
fotografiju pčele, sistem je prosledi modelu mašinskog učenja, a model vrati
procenu zdravstvenog stanja i verovatnoću te procene. Rezultat se vezuje za
konkretnu košnicu i čuva u istoriji pregleda.

Model se razvija **transfer learning**-om: umesto da se mreža trenira od nule,
koriste se mreže prethodno istrenirane na velikom opštem skupu slika, koje se
prilagođavaju ovom zadatku. Upoređuju se tri arhitekture — **MobileNetV3,
ResNet50 i EfficientNetB0** — pod istim uslovima, po metrikama accuracy,
precision, recall, F1-score i matrici konfuzije, i najbolja se integriše u
sistem.

## Istraživačka pitanja

Rad odgovara na sledeća pitanja:

1. Koji kvalitetni javni skupovi označenih slika pčela postoje, koje klase
   sadrže i da li odgovaraju ovom problemu?
2. Kako se transfer learning praktično primenjuje sa MobileNet, ResNet i
   EfficientNet modelima?
3. Da li je za ovaj zadatak pogodniji PyTorch ili TensorFlow/Keras?
4. Koja od tri arhitekture daje najbolje rezultate, i po kojim kriterijumima?
5. Kako se istrenirani model povezuje sa postojećim ASP.NET sistemom?

## Tok rada i ključni nalaz

Rad je izveden u dva eksperimenta. Prvi je koristio skup **BeeImage**, najpoznatiji
javni skup slika pčela sa oznakama zdravlja. Modeli su na njegovom test skupu
postigli tačnost do 97%, ali je ručno testiranje u aplikaciji pokazalo da model
fotografiju pčele sa dve jasno vidljive grinje procenjuje kao zdravu sa 99,97%.
Analiza je pokazala da oznake u tom skupu opisuju snimak, a ne pčelu, i da je
mreža naučila da prepoznaje pozadinu snimka umesto bolesti — pojava poznata kao
*shortcut learning*.

Drugi, glavni eksperiment koristi skup **VarroaDataset**, u kome je svaka pčela
označena posebno, na osnovu vidljive grinje *Varroa destructor*. Uz klasične
metrike, za svaki model je posebnim testovima provereno **da li zaista gleda
grinju** — što je pouka prvog eksperimenta.

## Struktura izveštaja

| Poglavlje | Sadržaj | Pitanje |
|---|---|---|
| Pregled javnih skupova podataka | pet skupova, kriterijumi, izbor | 1 |
| Prvi eksperiment: BeeImage | rezultati i zašto model ne radi | 1 |
| Skup podataka VarroaDataset | analiza izabranog skupa | 1 |
| Transfer learning — pristup i primena | postupak korak po korak | 2 |
| Izbor okvira: PyTorch ili TensorFlow/Keras | obrazloženje | 3 |
| Eksperimentalni rezultati | poređenje tri arhitekture | 4 |
| Da li model gleda grinju? | provere ponašanja modela | 4 |
| Test van skupa za trening | drugi skup i terenske fotografije | 4 |
| Izbor modela | odluka i obrazloženje | 4 |
| Integracija u sistem „Košnica“ | arhitektura i implementacija | 5 |
| Ograničenja i dalji rad | | |
