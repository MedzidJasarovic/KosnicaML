# Ograničenja i dalji rad

## Ograničenja skupa podataka

**Jedno okruženje snimanja.** Sve slike VarroaDataset-a snimljene su istom
opremom, odozgo, na istoj sivoj podlozi. Model je zato video samo jednu vrstu
pozadine i osvetljenja. Augmentacija (rotacija, osvetljenje, boja, zamućenje)
ublažava, ali ne rešava ovo ograničenje. Koliko se model prenosi na drugačije
snimke meri se na skupu BeeDataset i terenskim fotografijama (poglavlje „Test
van skupa za trening“).

**Dve klase.** Model razlikuje samo zdravu pčelu i pčelu sa vidljivom varoom.
Druga zdravstvena stanja (deformisana krila, druge bolesti, odsustvo matice)
nisu obuhvaćena, jer nijedan pronađeni javni skup nema pouzdane oznake za njih
po pojedinačnoj pčeli.

**Vidljiva grinja nije isto što i zaražena košnica.** Model procenjuje da li se
na fotografisanoj pčeli vidi grinja. Pčela bez vidljive grinje može poticati iz
zaražene košnice — grinja može biti ispod krila, između segmenata ili u leglu.
Za procenu stanja košnice potrebno je analizirati više pčela; pojedinačna
procena je pomoćna informacija pčelaru, a ne dijagnoza.

**Nedokumentovana oznaka.** Oznaka 3 u skupu nije opisana u dokumentaciji. U
radu se tretira kao zaražena pčela, u skladu sa zvaničnom statistikom autora;
rezultati su prikazani i odvojeno za oznake 1 i 3.

**Moguće preklapanje val i train snimka.** Dva snimka iz iste sesije (25.09.2017)
nalaze se jedan u trening, a drugi u validacionom skupu. To može blago ulepšati
validacione rezultate, na osnovu kojih se bira checkpoint i protokol, ali ne
utiče na test skup.

## Ograničenja eksperimenta

**Jedno seme.** Svaka arhitektura trenirana je jednom, sa semenom 42. Razlike
između arhitektura manje od oko jednog procentnog poena nisu nužno značajne —
za pouzdaniji zaključak trening bi trebalo ponoviti sa više semena i prikazati
srednju vrednost i rasipanje. Trening na MPS backend-u ni sa fiksnim semenom
nije potpuno determinističan.

**Protokol biran na jednoj arhitekturi.** Protokol treninga (fine-tuning cele
mreže, stopa učenja, veličina slike) izabran je na validacionom skupu sa
MobileNetV3, pa primenjen na sve tri. Druga arhitektura bi mogla imati koristi
od drugačijih parametara; izabran je jedan zajednički protokol da bi poređenje
bilo pod istim uslovima, kako je traženo.

**Mali uzorak terenskih fotografija.** Terenski test ima malo fotografija i
služi kao ilustracija, ne kao statistički pouzdana mera. Folder
`data/field_photos/` je pripremljen tako da se proširi i test ponovi.

**Grad-CAM je gruba mapa.** Poslednji konvolucioni sloj ima prostornu
rezoluciju 7×7 (odnosno 10×10 za veće ulaze), pa mapa pokazuje područje, ne
tačnu tačku. Zato je u merenju korišćena tolerancija, a glavni dokaz da model
gleda grinju je test brisanja grinje, ne Grad-CAM.

## Ograničenja integracije

**Skladištenje slika je lokalno.** Fotografije se čuvaju na disku servera
(`wwwroot/uploads/`). Za produkciju na platformi sa netrajnim diskom (npr.
Render) potrebno je trajno skladište (npr. Cloudinary ili S3) — interfejs
`IStorageService` je za to već pripremljen.

**Docker slika servisa nije testirana** na mašini na kojoj je rad izveden
(Docker nije instaliran); servis je testiran lokalno.

## Dalji rad

1. **Proširenje terenskog test skupa** fotografijama iz stvarnih pčelinjaka i
   ponovna procena. Ako se pokaže pad na drugačijim pozadinama: augmentacija
   zamenom pozadine (izdvajanje pčele sa sive podloge i postavljanje na
   raznovrsne pozadine) ili dodavanje BeeDataset-a u trening.
2. **Detekcija umesto klasifikacije.** Koordinate grinja u VarroaDataset-u
   omogućavaju treniranje detektora (npr. YOLO) koji pronalazi i broji grinje
   na fotografiji sa više pčela — bliže stvarnom pregledu okvira.
3. **Procena na nivou košnice.** Agregacija procena više pčela iz iste košnice
   u procenu stepena zaraze, uz praćenje kroz vreme u istoriji pregleda.
4. **Automatska intervencija.** Pri proceni varoe sa visokom verovatnoćom sistem
   može predložiti intervenciju tipa „Tretman“ sa statusom „Planirano“, koja se
   pojavljuje u postojećem planeru pčelara.
5. **Kalibracija praga odluke.** Prag 0,5 može se pomeriti ka većoj osetljivosti
   (više otkrivenih zaraza po cenu više lažnih uzbuna), u zavisnosti od toga šta
   je pčelaru skuplje.
