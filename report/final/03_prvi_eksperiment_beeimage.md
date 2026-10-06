# Prvi eksperiment: BeeImage

## Postavka

Prvi eksperiment je izveden nad skupom BeeImage (5.172 slike), sa svim šest
originalnih klasa i sa klasama grupisanim u tri: `zdrava`, `varoa_grinje`
(obe klase sa varoom) i `drugi_problem` (mravi, pljačka, bez matice). Slike su
nasumično, stratifikovano po klasi, podeljene u odnosu 70/15/15. Trenirane su
sve tri arhitekture istim postupkom transfer learning-a kao u glavnom
eksperimentu.

## Rezultati na test skupu

<!-- TABELA:beeimage_3cls -->

Po svim metrikama rezultati su izuzetno dobri. Najbolji model (MobileNetV3-Large)
izvezen je i integrisan u sistem.

## Problem otkriven ručnim testom

Pri ručnom testiranju u aplikaciji, fotografija pčele na saću sa dve jasno
vidljive grinje procenjena je kao **zdrava sa 99,97%**.

![Fotografija koja je otkrila problem: pčela sa dve vidljive grinje, model kaže „zdrava“ 99,97%](data/field_photos/varoa/makro_pcela_na_sacu_2_grinje.jpg){width=40%}

## Uzrok: oznaka opisuje snimak, ne pčelu

Analiza skupa pokazala je tri svojstva koja zajedno objašnjavaju ponašanje modela:

1. **Oznaka je dodeljena celom snimku.** BeeImage je nastao isecanjem pčela iz
   video snimaka na ulazu u košnicu; svaka pčela dobija oznaku stanja košnice.
   Na većini isečaka iz klase „varoa“ grinja se ne vidi.
2. **Svaki snimak pripada tačno jednoj klasi.** Postoji 45 snimaka (prepoznatih
   po prefiksu imena fajla) i nijedan ne prelazi granicu klase. Klasa varoe
   potiče iz svega pet snimaka, a 89% njenih slika iz dva.
3. **Nasumična podela stavlja isti snimak i u trening i u test.** Uzastopni
   frejmovi istog videa su skoro identični, pa test skup sadrži „iste“ slike kao
   trening.

<!-- TABELA:beeimage_snimci -->

Mreža se trenira tako da što manje greši, i koristi bilo koju razliku između
klasa koja joj u tome pomaže. Najlakša takva razlika u ovom skupu je pozadina,
osvetljenje i podloga pojedinog snimka — ne izgled pčele.

![Primeri slika po klasi u skupu BeeImage — klasa varoe ima prepoznatljivu tirkiznu podlogu, a grinja se ne vidi](report/figures/samples_per_class.png){width=85%}

## Dokaz: test prečice

Da bi se hipoteza proverila, modelu su date samo četiri ugla svake test slike
(25% širine i 25% visine). Pčela je u centru isečka, pa uglovi sadrže pozadinu.
Kada bi model gledao pčelu, iz uglova bi pogađao na nivou slučaja (33% za tri
klase).

| Prava klasa | Slika | Cela slika — tačno | Samo uglovi — tačno |
|---|---|---|---|
| zdrava | 507 | 96,4% | 84,0% |
| varoa_grinje | 158 | 97,5% | **73,9%** |
| drugi_problem | 111 | 99,1% | 10,8% |

Iz komadića pozadine na kome pčele nema, model u 74% slučajeva „prepoznaje“
varou. Tačnost od 97% je dakle tačan broj, ali meri prepoznavanje snimka, a ne
bolesti. Model nije upotrebljiv za stvarnu procenu zdravlja pčele.

Reprodukcija: `python -m src.shortcut_test`.

## Pouke za glavni eksperiment

1. Skup mora imati **oznaku po pčeli**, dodeljenu na osnovu vidljivog obeležja.
2. Pozadina mora biti **ista za sve klase**.
3. Podela mora biti **po snimku**, tako da test sadrži snimke koje model nije video.
4. Visoka tačnost na test skupu nije dovoljna — treba **proveriti šta model
   gleda**, i testirati ga na slikama drugačijeg porekla.

Sva četiri zahteva su ugrađena u glavni eksperiment.
