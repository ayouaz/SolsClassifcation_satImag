# Guide de fine-tuning — Classification de sols par images satellitaires

Ce guide explique comment utiliser `finetune.py` pour adapter un Vision Transformer (ViT) à la classification de sols à partir d'images satellitaires multibandes (Sentinel-2, Landsat, drone…).

---

## 1. Prérequis

### Matériel recommandé

- GPU NVIDIA avec CUDA (ex. RTX 3090, A100) — 16 Go de VRAM minimum pour ViT-Base.
- CPU uniquement possible pour des tests (lent).

### Installation

```bash
pip install torch torchvision pytorch-lightning
pip install transformers rasterio albumentations
```

Ou via le `requirements.txt` du backend qui inclut toutes ces dépendances.

---

## 2. Structure du dataset

```
dataset/
├── train/
│   ├── classe1/   ← ex. "sol_ferrugineux"
│   │   ├── patch_001.tif
│   │   └── ...
│   └── classe2/   ← ex. "sol_argileux"
│       └── ...
├── val/
│   ├── classe1/
│   └── classe2/
└── test/          ← optionnel
    ├── classe1/
    └── classe2/
```

Chaque image doit être un GeoTIFF de **224 × 224 pixels** (taille d'entrée du ViT).
Pour des images plus grandes, découpez-les en patches avec `rasterio` ou `TorchGeo`.

**Important :** le label est déduit du nom du dossier parent. Le script extrait le dernier caractère du nom de dossier comme entier : `class1 → 0`, `class2 → 1`, etc. Pour des classes avec des noms personnalisés, adaptez la fonction `label_from_path()` dans `finetune.py`.

---

## 3. Bandes spectrales supportées

Le modèle s'adapte automatiquement au nombre de bandes fourni via le paramètre `num_bands`.

| `num_bands` | Bandes typiques | Source |
|-------------|----------------|--------|
| `3` | B, G, R | RGB standard |
| `4` | B, G, R, NIR | Sentinel-2 (B2,B3,B4,B8) / Landsat |
| `10` | B2–B8, B8A, B11, B12 | Sentinel-2 SR (10 bandes) |
| `13` | Toutes | Sentinel-2 SR complet |

Pour tout autre nombre de bandes, des stats génériques (mean=0.5, std=0.25) sont utilisées automatiquement. Des stats calculées sur votre dataset donneront de meilleurs résultats.

---

## 4. Utilisation

### Lancement rapide (données factices)

```python
from finetune import main
main(num_bands=4, num_classes=2, max_epochs=10)
```

### Sur vos propres données

```python
import glob, os
from finetune import SatelliteDataset, SatelliteClassifier, get_norm_stats
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch

NUM_BANDS   = 10   # adapter à vos images
NUM_CLASSES = 5    # nombre de classes de sols

mean, std = get_norm_stats(NUM_BANDS)

train_transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])

train_paths = glob.glob("dataset/train/**/*.tif", recursive=True)
val_paths   = glob.glob("dataset/val/**/*.tif",   recursive=True)

def label_from_path(p):
    return int(os.path.basename(os.path.dirname(p))[-1]) - 1

train_ds = SatelliteDataset(train_paths, [label_from_path(p) for p in train_paths], NUM_BANDS, train_transform)
val_ds   = SatelliteDataset(val_paths,   [label_from_path(p) for p in val_paths],   NUM_BANDS, val_transform)

model = SatelliteClassifier(num_classes=NUM_CLASSES, num_bands=NUM_BANDS, lr=1e-4)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = Trainer(max_epochs=20, accelerator=accelerator, devices=1)
trainer.fit(model, DataLoader(train_ds, batch_size=8, shuffle=True), DataLoader(val_ds, batch_size=8))
```

---

## 5. Architecture du modèle

Le `SatelliteClassifier` adapte un ViT pré-entraîné (`google/vit-base-patch16-224-in21k`) :

```
Images (B, num_bands, 224, 224)
  └── Conv2d(num_bands, 768, 16×16)   ← patch embedding adapté
       └── 12 blocs Transformer        ← poids pré-entraînés ImageNet
            └── Token [CLS] (768-dim)
                 └── Linear(768, num_classes)
                      └── CrossEntropyLoss
```

**Adaptation multibandes :** les 3 premiers canaux reprennent les poids RGB pré-entraînés ; les canaux supplémentaires (NIR, SWIR…) sont initialisés avec la moyenne des poids RGB. Cela permet de démarrer l'entraînement avec un modèle déjà partiellement convergé plutôt que de partir de zéro.

**Optimiseur :** AdamW avec weight decay 1e-4, lr par défaut 1e-4 (ajustable).

---

## 6. Inférence sur une nouvelle image

```python
import torch
from finetune import SatelliteClassifier, load_tiff, get_norm_stats
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_BANDS   = 4
NUM_CLASSES = 2

# Charger le checkpoint
model = SatelliteClassifier.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=9-step=10.ckpt"
)
model.eval()

mean, std = get_norm_stats(NUM_BANDS)
transform = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])

img = load_tiff("nouvelle_image.tif", num_bands=NUM_BANDS)
tensor = transform(image=img)["image"].unsqueeze(0)  # (1, C, H, W)

with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)
    classe = torch.argmax(probs, dim=1).item()

print(f"Classe prédite : {classe} | Probabilités : {probs.squeeze().tolist()}")
```

---

## 7. Métriques de suivi

Les métriques suivantes sont loggées automatiquement dans TensorBoard :

| Métrique | Description |
|----------|-------------|
| `train_loss` | Cross-entropy sur le batch d'entraînement |
| `val_loss` | Cross-entropy sur la validation |
| `val_acc` | Accuracy sur la validation |

Lancer TensorBoard :
```bash
tensorboard --logdir lightning_logs/
```

---

## 8. Conseils pratiques

- **Nombre de bandes :** utilisez toutes les bandes disponibles ; les bandes SWIR (B11, B12 pour Sentinel-2) sont particulièrement discriminantes pour les sols et les minéraux ferreux.
- **Taille du dataset :** minimum ~200 images par classe pour un fine-tuning stable. Au-dessous, utilisez une augmentation agressive et un learning rate plus faible (1e-5).
- **Transfer learning :** geler les couches Transformer (`self.vit.requires_grad_(False)`) et n'entraîner que le patch embedding adapté + la tête de classification pendant les premières époques peut accélérer la convergence.
- **Normalisation :** si vos valeurs de réflectance sont déjà dans [0, 1], passez `max_pixel_value=1.0` à `A.Normalize`.
- **GPU :** le script détecte automatiquement la présence d'un GPU CUDA.

---

## 9. Datasets publics pour démarrer

| Dataset | Bandes | Classes | Taille |
|---------|--------|---------|--------|
| [EuroSAT](https://github.com/phelber/EuroSAT) | 13 (Sentinel-2) | 10 | 27 000 images |
| [BigEarthNet](https://bigearth.net) | 12 (Sentinel-2) | 43 | 590 000 images |
| [UC Merced Land Use](http://weegee.vision.ucmerced.edu/datasets/landuse.html) | 3 (RGB) | 21 | 2 100 images |

---

## 10. Outils complémentaires

- **[TorchGeo](https://github.com/microsoft/torchgeo)** : datasets géospatiaux prêts à l'emploi pour PyTorch, incluant EuroSAT et BigEarthNet.
- **[Raster Vision](https://docs.rastervision.io)** : framework complet pour la détection et la segmentation sur images satellitaires.
- **[SatMAE](https://github.com/sustainlab-group/SatMAE)** : ViT pré-entraîné spécifiquement sur des images satellitaires multibandes (meilleure alternative à `vit-base-patch16-224-in21k` pour ce domaine).
