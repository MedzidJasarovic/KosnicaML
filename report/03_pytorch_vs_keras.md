# Izbor okvira: PyTorch ili TensorFlow/Keras

Oba okvira podržavaju sve tri tražene arhitekture sa ImageNet težinama i oba su
u potpunosti dovoljna za ovaj rad. Odabran je **PyTorch sa `torchvision`**
bibliotekom, iz tri razloga specifična za prirodu ovog zadatka.

**1. Preprocesiranje je vezano za težine, ne za programera.** U `torchvision`-u
svaki objekat prethodno istreniranih težina nosi i metodu `transforms()` koja
vraća tačno ono preprocesiranje sa kojim je model treniran. U Keras-u svaka
arhitektura ima zasebnu funkciju `preprocess_input` sa različitim konvencijama —
`EfficientNet` u Keras-u očekuje vrednosti piksela u opsegu 0–255 i normalizaciju
izvodi interno, dok `ResNet50` očekuje već skalirane vrednosti. Zamena tih
funkcija ne izaziva grešku u izvršavanju, samo tiho pogoršava rezultat.
Pošto je cilj ovog rada **poređenje tri arhitekture pod što sličnijim uslovima**,
okvir u kojem je ta klasa greške strukturno onemogućena ima direktnu
metodološku prednost.

**2. Jedna petlja treninga za sve tri arhitekture.** U PyTorch-u se petlja
treninga piše eksplicitno, pa su optimizator, stopa učenja, raspored opadanja,
broj epoha, kriterijum izbora checkpointa i obrada disbalansa doslovno isti kod
za sva tri modela — razlikuje se samo telo mreže. Eksplicitna petlja je ovde
prednost, a ne režijski trošak: čini uslove eksperimenta proverljivim.

**3. Čist put do produkcije.** Model se izvozi u ONNX direktno iz PyTorch-a
(`torch.onnx.export`), bez posrednog koraka. Iz Keras-a je potrebna dodatna
konverzija alatom `tf2onnx`.

Dodatno, PyTorch ima nativnu podršku za **MPS** backend, što omogućava trening
na grafičkom procesoru Apple Silicon računara na kojem je rad izveden, bez
oslanjanja na eksterne GPU servise.

Verzije korišćene u radu: Python 3.12.15, PyTorch 2.14.1, torchvision 0.29.1,
scikit-learn 1.9.1, onnxruntime 1.30.0.
