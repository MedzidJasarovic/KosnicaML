# Integracija modela u informacioni sistem „Košnica"

## Postojeći sistem

| Sloj | Tehnologija |
|---|---|
| Frontend | Angular 17, standalone komponente, TypeScript |
| Backend | ASP.NET Core (.NET 10), REST API, JWT autentifikacija |
| Baza | PostgreSQL, Entity Framework Core |
| Eksterne integracije | OpenWeatherMap |

## Predložena arhitektura

Model se ne izvršava unutar .NET procesa, nego u zasebnom Python servisu.
.NET backend ostaje jedina tačka koju frontend poznaje i jedini sloj koji
pristupa bazi i proverava autorizaciju.

```
Angular (hive-detail modal, tab "Analiza fotografije")
   │  POST /api/predictions/bee-health   (multipart: slika + hiveId, JWT u header-u)
   ▼
ASP.NET Core — PredictionsController
   │  provera vlasništva nad košnicom, snimanje slike, upis rezultata u bazu
   │
   ├─► IStorageService ──────────► trajno skladište slike
   │
   ├─► IBeeHealthService ────────► HTTP POST /predict
   │                                   ▼
   │                            FastAPI servis (Python)
   │                            onnxruntime + bee_health.onnx
   │                            preprocesiranje po bee_health.meta.json
   │                                   │
   │                            { label, confidence, all_probs }
   │                                   ▼
   └─► PostgreSQL: tabela BeeHealthPredictions (istorija pregleda)
```

## Zašto zaseban Python servis, a ne model unutar .NET-a

Razmatrane su dve opcije.

**Opcija A — ONNX Runtime unutar .NET-a** (`Microsoft.ML.OnnxRuntime`).
Prednost: jedan proces, jedan deployment. Nedostatak: preprocesiranje slike
(skaliranje kraće strane, centralni crop, normalizacija po kanalu) moralo bi se
ručno reimplementirati u C#. Svako odstupanje od preprocesiranja iz treninga
tiho pogoršava predikcije, bez greške u izvršavanju.

**Opcija B — Python/FastAPI servis** (izabrano). Preprocesiranje u servisu
koristi iste biblioteke i iste parametre kao trening, pročitane iz
`bee_health.meta.json` koji je nastao pri izvozu modela. Rizik tihog odstupanja
je time eliminisan. Servis je i nezavisno skalabilan, a model se može zameniti
novom verzijom bez ponovnog deploy-a .NET aplikacije.

Model se izvozi u **ONNX**, pa servis koristi `onnxruntime` umesto PyTorch-a —
Docker slika je time reda 100 MB, a ne nekoliko gigabajta. Isti `.onnx` fajl
ostaje upotrebljiv i za Opciju A, ako bi se kasnije prešlo na jedan stek.

## Potrebne promene u backendu

| Komponenta | Opis |
|---|---|
| `Services/IBeeHealthService.cs` + `BeeHealthService.cs` | HTTP klijent ka FastAPI servisu. Po uzoru na postojeći `OpenWeatherMapService`, uključujući njegov obrazac za slučaj nedostupnog servisa. |
| `Models/BeeHealthPrediction.cs` | Nova tabela: `HiveId`, `ImageUrl`, `PredictedLabel`, `Confidence`, `ModelVersion`, `CreatedAt`, `CreatedByUserId`. |
| `Data/ApplicationDbContext.cs` | Novi `DbSet` i EF migracija. Migracije se primenjuju automatski pri pokretanju aplikacije. |
| `Controllers/PredictionsController.cs` | `POST /api/predictions/bee-health` i `GET /api/predictions/hive/{hiveId}` (istorija pregleda po košnici). |

Adresa ML servisa čita se iz promenljive okruženja, po već postojećem obrascu
za konekcioni string baze — ne upisuje se u `appsettings.json`.

Opciono: pri detekciji varoe sa visokom verovatnoćom sistem može automatski
kreirati `Intervention` tipa `Treatment` sa statusom `Planned`, čime predikcija
postaje konkretan zadatak u planeru pčelara.

## Potrebne promene u frontendu

Novi `services/prediction.service.ts` po uzoru na postojeći `hive.service.ts`.
U modalu detalja košnice (`hive-detail-modal`) dodaje se treći tab —
komponenta već koristi obrazac sa `activeTab`. Tab sadrži izbor fajla, prikaz
rezultata kao obojene oznake sa procentom pouzdanosti, i listu prethodnih
analiza za tu košnicu.

## Zavisnost koju treba rešiti pre integracije

Trenutna implementacija skladištenja u sistemu je privremena: servis
`MockStorageService` ne čuva fajl, nego vraća generisan URL, pa u bazu ulazi
samo string. Takođe, u Angular aplikaciji trenutno ne postoji komponenta za
upload fajlova.

Funkcionalnost analize fotografije zato podrazumeva i implementaciju stvarnog
skladištenja (najjednostavnije: lokalni folder posluživan kao statički sadržaj)
i izradu upload komponente. To je deo obima integracije, nezavisan od modela.
