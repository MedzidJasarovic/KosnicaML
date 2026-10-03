# Eksperimentalni rezultati

## Uslovi eksperimenta

Sve arhitekture trenirane su pod identičnim uslovima; razlikuje se
isključivo telo mreže.

| Parametar | Vrednost |
|---|---|
| Podela podataka | stratifikovano 70/15/15, isti fajlovi za sve modele |
| Ulazna rezolucija | prema `weights.transforms()` svake arhitekture |
| Veličina batch-a | 32 |
| Faza 1 (glava) | 15 epoha, telo zamrznuto, lr 1e-3 |
| Faza 2 (fine-tuning) | 10 epoha, poslednji blok odmrznut, lr 1e-5 |
| Optimizator | AdamW, weight decay 1e-4 |
| Raspored stope učenja | CosineAnnealingLR |
| Funkcija greške | CrossEntropyLoss sa težinama klasa |
| Augmentacija | RandomResizedCrop, flip, rotacija 15°, ColorJitter 0.2 |
| Izbor checkpointa | najveći macro F1 na validacionom skupu |
| Seed | 42 |
| Uređaj | Apple M5, PyTorch MPS |

## Poređenje arhitektura — 6 klasa

Test skup: 776 slika.

| Arhitektura | Accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall | Recall (varoa) | Parametara | Latencija (ms) | Trening (min) |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large | 0.9124 | 0.8031 | 0.9153 | 0.7753 | 0.8388 | 0.9684 | 4.2M | 41.1 | 3.3 |
| ResNet50 | 0.9072 | 0.8218 | 0.9103 | 0.7796 | 0.9032 | 0.9494 | 23.5M | 12.4 | 11.8 |
| EfficientNetB0 | 0.8273 | 0.7020 | 0.8334 | 0.6442 | 0.8200 | 0.9304 | 4.0M | 84.2 | 5.8 |

### MobileNetV3-Large — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.994 | 0.925 | 0.958 | 507 |
| few_varroa_hive_beetles | 0.766 | 0.828 | 0.796 | 87 |
| varroa_small_hive_beetles | 0.753 | 0.859 | 0.803 | 71 |
| ant_problems | 0.793 | 1.000 | 0.885 | 69 |
| hive_being_robbed | 0.946 | 0.921 | 0.933 | 38 |
| missing_queen | 0.400 | 0.500 | 0.444 | 4 |

![Confusion matrix](runs/mobilenet_v3_large_6cls/confusion_matrix.png)

![Krive učenja](runs/mobilenet_v3_large_6cls/learning_curves.png)

### ResNet50 — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.985 | 0.927 | 0.955 | 507 |
| few_varroa_hive_beetles | 0.742 | 0.793 | 0.767 | 87 |
| varroa_small_hive_beetles | 0.750 | 0.845 | 0.795 | 71 |
| ant_problems | 0.840 | 0.986 | 0.907 | 69 |
| hive_being_robbed | 0.917 | 0.868 | 0.892 | 38 |
| missing_queen | 0.444 | 1.000 | 0.615 | 4 |

![Confusion matrix](runs/resnet50_6cls/confusion_matrix.png)

![Krive učenja](runs/resnet50_6cls/learning_curves.png)

### EfficientNetB0 — metrike po klasi (6 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| healthy | 0.979 | 0.844 | 0.907 | 507 |
| few_varroa_hive_beetles | 0.603 | 0.506 | 0.550 | 87 |
| varroa_small_hive_beetles | 0.573 | 0.887 | 0.696 | 71 |
| ant_problems | 0.731 | 0.986 | 0.840 | 69 |
| hive_being_robbed | 0.679 | 0.947 | 0.791 | 38 |
| missing_queen | 0.300 | 0.750 | 0.429 | 4 |

![Confusion matrix](runs/efficientnet_b0_6cls/confusion_matrix.png)

![Krive učenja](runs/efficientnet_b0_6cls/learning_curves.png)

## Poređenje arhitektura — 3 klasa

Test skup: 776 slika.

| Arhitektura | Accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall | Recall (varoa) | Parametara | Latencija (ms) | Trening (min) |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large | 0.9704 | 0.9643 | 0.9706 | 0.9529 | 0.9767 | 0.9747 | 4.2M | 40.9 | 3.3 |
| ResNet50 | 0.9716 | 0.9625 | 0.9718 | 0.9557 | 0.9698 | 0.9620 | 23.5M | 12.7 | 12.7 |
| EfficientNetB0 | 0.9111 | 0.8954 | 0.9124 | 0.8701 | 0.9278 | 0.9241 | 4.0M | 85.4 | 5.7 |

### MobileNetV3-Large — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.994 | 0.965 | 0.979 | 507 |
| varoa_grinje | 0.917 | 0.975 | 0.945 | 158 |
| drugi_problem | 0.948 | 0.991 | 0.969 | 111 |

![Confusion matrix](runs/mobilenet_v3_large_3cls/confusion_matrix.png)

![Krive učenja](runs/mobilenet_v3_large_3cls/learning_curves.png)

### ResNet50 — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.988 | 0.974 | 0.981 | 507 |
| varoa_grinje | 0.956 | 0.962 | 0.959 | 158 |
| drugi_problem | 0.923 | 0.973 | 0.947 | 111 |

![Confusion matrix](runs/resnet50_3cls/confusion_matrix.png)

![Krive učenja](runs/resnet50_3cls/learning_curves.png)

### EfficientNetB0 — metrike po klasi (3 klasa)

| Klasa | Precision | Recall | F1 | Primera u testu |
|---|---|---|---|---|
| zdrava | 0.970 | 0.895 | 0.931 | 507 |
| varoa_grinje | 0.830 | 0.924 | 0.874 | 158 |
| drugi_problem | 0.811 | 0.964 | 0.881 | 111 |

![Confusion matrix](runs/efficientnet_b0_3cls/confusion_matrix.png)

![Krive učenja](runs/efficientnet_b0_3cls/learning_curves.png)
