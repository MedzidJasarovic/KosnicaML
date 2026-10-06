"""
Tri arhitekture iz mejla profesorice, pripremljene za transfer learning.

Transfer learning u tri poteza (isto za sve tri arhitekture):
  1. ucitaj model pretreniran na ImageNet-u,
  2. zameni zavrsni klasifikacioni sloj brojem klasa naseg problema,
  3. zamrzni telo (backbone) -> trenira se samo nova glava;
     kasnije se poslednji blok tela odmrzne za fine-tuning.

Preprocesiranje se NE pise rucno: svaki `Weights` objekat u torchvision-u nosi
`transforms()` sa tacnim resize/crop/normalizacijom sa kojom je model treniran.
Tako je nemoguce slucajno uporediti modele pod razlicitim uslovima.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.transforms import v2

from service.preprocessing import PAD_COLOR, letterbox

# ime -> (konstruktor, pretrenirani weights, naziv za izvestaj)
SPECS = {
    "mobilenet_v3_large": (
        tvm.mobilenet_v3_large,
        tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "MobileNetV3-Large",
    ),
    "resnet50": (
        tvm.resnet50,
        tvm.ResNet50_Weights.IMAGENET1K_V2,
        "ResNet50",
    ),
    "efficientnet_b0": (
        tvm.efficientnet_b0,
        tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "EfficientNetB0",
    ),
}

ARCHITECTURES = list(SPECS)


def _replace_head(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """
    Zamena zavrsnog sloja. Mesto se razlikuje po arhitekturi:
      ResNet        -> model.fc            (Linear)
      MobileNet/EffNet -> model.classifier[-1] (Linear u Sequential-u)
    Novi sloj je inicijalizovan slucajno i jedini je koji se u fazi 1 trenira.
    """
    if arch == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def head_parameters(model: nn.Module, arch: str):
    """Parametri nove glave — jedini koji se treniraju u fazi 1."""
    return (model.fc if arch == "resnet50" else model.classifier).parameters()


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Zamrzava sve, pa odmrzava samo novu glavu (faza 1 — feature extraction)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in head_parameters(model, arch):
        param.requires_grad = True


def unfreeze_last_block(model: nn.Module, arch: str) -> None:
    """
    Odmrzava poslednji konvolucioni blok (faza 2 — fine-tuning).
    Glava ostaje odmrznuta. Ostatak tela ostaje zamrznut: nizi slojevi su
    opste-vizualni (ivice, teksture) i ne treba ih menjati na 5k slika.
    """
    if arch == "resnet50":
        blocks = [model.layer4]
    else:
        # MobileNetV3 i EfficientNetB0: features je Sequential blokova
        blocks = [model.features[-1], model.features[-2]]

    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


class Letterbox:
    """PIL -> PIL letterbox; klasa (ne lambda) da bi se mogla poslati DataLoader radnicima."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        return letterbox(image, self.size)


class LetterboxEval:
    """
    Eval transform za VarroaDataset: letterbox na kvadrat, pa normalizacija.
    Ima iste atribute kao torchvision `weights.transforms()` (crop_size,
    resize_size, mean, std), pa ga ostatak koda koristi na isti način.
    Sama priprema slike je u service/preprocessing.py — ista funkcija radi i u servisu.
    """

    def __init__(self, size: int, mean, std):
        self.crop_size = [size]
        self.resize_size = [size]
        self.mean = list(mean)
        self.std = list(std)
        self._to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, image):
        return self._to_tensor(letterbox(image, self.crop_size[0]))


def _letterbox_transforms(weights, size: int | None = None) -> tuple[object, object]:
    """
    (train_tf, eval_tf) za VarroaDataset.

    Slike su uspravne (160x280). Center crop iz weights.transforms() bi odsekao
    ~45% visine — glavu i zadak, gde je grinja često. Zato se cela slika smanji
    i dopuni do kvadrata. Augmentacija je jača nego za BeeImage, jer model treba
    da radi i na fotografijama korisnika: rotacija u svim smerovima (pčela je
    snimljena odozgo, pa nema "gore"), promena razmere, boje, osvetljenja i
    zamućenje. Nijansa (hue) se menja malo, jer je crvenkasta boja grinje signal.
    """
    ref = weights.transforms()
    crop = size or (ref.crop_size[0] if isinstance(ref.crop_size, (list, tuple)) else ref.crop_size)
    eval_tf = LetterboxEval(crop, ref.mean, ref.std)

    train_tf = v2.Compose([
        Letterbox(crop),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.8, 1.1), fill=list(PAD_COLOR)),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=ref.mean, std=ref.std),
    ])
    return train_tf, eval_tf


def unfreeze_all(model: nn.Module) -> None:
    """
    Odmrzava celu mrežu (faza 2 — potpun fine-tuning). Potrebno kada je obeležje
    koje se traži sitno i van ImageNet domena (grinja od ~25 px): zamrznuti
    niži slojevi ga ne izdvajaju dovoljno dobro.
    """
    for param in model.parameters():
        param.requires_grad = True


def build_model(arch: str, num_classes: int, preprocessing: str = "center_crop",
                img_size: int | None = None) -> tuple[nn.Module, object, object]:
    """
    Vraca (model, train_transform, eval_transform).

    preprocessing="center_crop" (BeeImage): eval_transform je zvanicni
    `weights.transforms()`, train_transform isti pipeline + augmentacija.
    preprocessing="letterbox" (VarroaDataset): vidi _letterbox_transforms.
    U oba slucaja su transformacije identicne za sve tri arhitekture.
    """
    if arch not in SPECS:
        raise ValueError(f"Nepoznata arhitektura '{arch}'. Dostupne: {ARCHITECTURES}")

    constructor, weights, _ = SPECS[arch]
    model = constructor(weights=weights)
    model = _replace_head(model, arch, num_classes)

    if preprocessing == "letterbox":
        train_tf, eval_tf = _letterbox_transforms(weights, img_size)
        return model, train_tf, eval_tf

    eval_tf = weights.transforms()
    crop = eval_tf.crop_size[0] if isinstance(eval_tf.crop_size, (list, tuple)) else eval_tf.crop_size

    train_tf = v2.Compose([
        v2.RandomResizedCrop(crop, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=eval_tf.mean, std=eval_tf.std),
    ])

    return model, train_tf, eval_tf


def display_name(arch: str) -> str:
    return SPECS[arch][2]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(ukupno, trenirajucih) — ide u tabelu poredjenja u izvestaju."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def pick_device() -> torch.device:
    """MPS na Apple Silicon, CUDA ako postoji, inace CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
