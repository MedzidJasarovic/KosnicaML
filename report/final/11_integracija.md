# Integracija u informacioni sistem „Košnica“

## Postojeći sistem

| Sloj | Tehnologija |
|---|---|
| Frontend | Angular 17, samostalne (*standalone*) komponente, TypeScript |
| Backend | ASP.NET Core (.NET 10), REST API, JWT autentifikacija |
| Baza | PostgreSQL, Entity Framework Core |
| Spoljne integracije | OpenWeatherMap |

## Arhitektura rešenja

Model se ne izvršava unutar .NET procesa, nego u zasebnom **Python servisu
(FastAPI)**. ASP.NET backend ostaje jedina tačka koju frontend poznaje i jedini
sloj koji pristupa bazi i proverava ovlašćenja.

```
Angular — modal košnice, tab „Zdravlje pčela“
   │  POST /api/hives/{hiveId}/health-checks   (multipart: slika, JWT)
   ▼
ASP.NET Core — BeeHealthChecksController
   │  1. provera da korisnik ima pravo na košnicu
   │  2. provera fajla (format, veličina)
   │  3. BeeHealthService ──HTTP POST /predict──► FastAPI servis (Python)
   │                                              onnxruntime + bee_health.onnx
   │                                              priprema slike po bee_health.meta.json
   │                      ◄── {label, confidence, all_probs, model_version}
   │  4. tek ako je analiza uspela: slika se snima (LocalStorageService)
   │  5. upis u tabelu BeeHealthChecks — istorija pregleda
   ▼
JSON odgovor → Angular prikazuje procenu, verovatnoće i istoriju
```

## Zašto zaseban Python servis

Razmatrane su dve opcije.

**Opcija A — ONNX Runtime unutar .NET-a** (`Microsoft.ML.OnnxRuntime`).
Prednost: jedan proces i jedan deployment. Nedostatak: priprema slike morala bi
se ponovo napisati u C#. Svako odstupanje od pripreme iz treninga tiho
pogoršava predikcije, bez ikakve greške u izvršavanju.

**Opcija B — Python/FastAPI servis** (izabrano). Servis koristi istu funkciju
za pripremu slike kao trening (`service/preprocessing.py`), sa parametrima
pročitanim iz `bee_health.meta.json` koji nastaje pri izvozu modela. Rizik tihog
odstupanja je time uklonjen. Servis se može nezavisno skalirati, a model
zameniti novom verzijom bez ponovnog objavljivanja .NET aplikacije — upravo to
je u ovom radu i urađeno pri prelasku sa BeeImage na VarroaDataset model.

Model se izvozi u format **ONNX**, pa servis koristi `onnxruntime` umesto
PyTorch-a: servis ne zavisi od biblioteke za trening, a Docker slika je reda
veličine 100 MB umesto nekoliko gigabajta.

## Python servis (FastAPI)

FastAPI je Python biblioteka za izradu REST API-ja, po ulozi slična ASP.NET
Minimal API-ju. Servis (`KosnicaML/service/`) ima dve rute:

| Ruta | Opis |
|---|---|
| `GET /health` | status, verzija modela, arhitektura, klase |
| `POST /predict` | prima sliku (multipart polje `file`), vraća procenu |

Primer odgovora:

```json
{
  "label": "varoa",
  "confidence": 0.9731,
  "all_probs": {"zdrava": 0.0269, "varoa": 0.9731},
  "model_version": "bee_health"
}
```

Model se učitava jednom, pri pokretanju servisa. Fajl koji nije slika vraća
grešku 400, fajl veći od 10 MB grešku 413.

**Provera ispravnosti.** Skripta `service/verify_against_pytorch.py` propušta
ceo test skup kroz servis (ONNX + priprema slike servisa) i kroz originalni
PyTorch model iz treninga, i poredi verovatnoće za svaku sliku.

<!-- TABELA:verifikacija_servisa -->

## Backend (ASP.NET Core)

| Komponenta | Opis |
|---|---|
| `Services/IBeeHealthService.cs`, `BeeHealthService.cs` | HTTP klijent ka FastAPI servisu; adresa iz promenljive okruženja `ML_SERVICE_URL`, inače `MlService:BaseUrl` iz `appsettings.json` |
| `Services/LocalStorageService.cs` | snima fajlove u `wwwroot/uploads/` i vraća njihov URL; zamenio je ranije postojeći `MockStorageService`, koji nije čuvao fajlove |
| `Models/BeeHealthCheck.cs` | nova tabela: košnica, URL slike, predviđena klasa, verovatnoća, verovatnoće svih klasa (`jsonb`), verzija modela, vreme, korisnik |
| `Controllers/BeeHealthChecksController.cs` | rute za analizu, istoriju i brisanje |
| EF migracija `AddBeeHealthChecks` | primenjuje se automatski pri pokretanju |

| Ruta | Opis | Ovlašćenje |
|---|---|---|
| `POST /api/hives/{hiveId}/health-checks` | analiza fotografije | svaki član tima pčelara |
| `GET /api/hives/{hiveId}/health-checks` | istorija pregleda, najnovije prvo | svaki član tima |
| `DELETE /api/hives/{hiveId}/health-checks/{id}` | brisanje pregleda i slike | samo pčelar (uloga *Beekeeper*) |

Ponašanje u greškama: ako ML servis nije dostupan, backend vraća 503 sa
porukom, a slika se ne snima. Pristup tuđoj košnici vraća 404 (ne otkriva se ni
da košnica postoji). Fajl koji nije slika vraća 400.

Svi navedeni slučajevi su provereni stvarnim HTTP zahtevima nad pokrenutim
sistemom.

## Frontend (Angular)

U modalu detalja košnice dodat je treći tab, **„Zdravlje pčela“**
(komponenta `hive-health-check`, servis `bee-health.service.ts`):

- izbor fotografije i pregled pre slanja
- rezultat: oznaka u boji (zelena — zdrava, crvena — varoa), verovatnoća
  procene i trake verovatnoća za svaku klasu
- istorija pregleda košnice sa sličicom, procenom, datumom i korisnikom
- brisanje pregleda uz potvrdu
- uputstvo korisniku: jedna pčela, izbliza, po mogućstvu odozgo

<!-- SLIKA:screenshot_aplikacije -->

## Zamena modela

Model se menja zamenom dva fajla — `models/bee_health.onnx` i
`models/bee_health.meta.json` — i ponovnim pokretanjem servisa. Backend i
frontend ne zavise od broja ni naziva klasa: backend čuva verovatnoće kao JSON
objekat, a frontend prikazuje klase koje model vrati. Stari zapisi u istoriji
(iz prvog, BeeImage modela, sa tri klase) i dalje se ispravno prikazuju.
