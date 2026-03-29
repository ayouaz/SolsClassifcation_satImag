import os
import glob
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from transformers import ViTModel
from pytorch_lightning import Trainer

# ---------------------------------------------------------------------------
# Stats de normalisation par nombre de bandes
# Sources : EuroSAT (Sentinel-2), ImageNet (RGB)
# Valeurs pour des images dont les réflectances sont dans [0, 1] après
# normalisation par max_pixel_value dans albumentations.
# ---------------------------------------------------------------------------
BAND_NORM_STATS: dict[int, dict] = {
    3: {  # RGB
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
    4: {  # RGB + NIR (ex. Sentinel-2 B2,B3,B4,B8)
        "mean": [0.485, 0.456, 0.406, 0.350],
        "std":  [0.229, 0.224, 0.225, 0.180],
    },
    10: {  # Sentinel-2 : B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
        "mean": [0.0911, 0.0895, 0.0809, 0.1021, 0.2226, 0.2627, 0.2503, 0.2324, 0.1024, 0.0560],
        "std":  [0.0530, 0.0573, 0.0638, 0.0745, 0.1091, 0.1265, 0.1188, 0.1120, 0.0561, 0.0431],
    },
    13: {  # Sentinel-2 : toutes les 13 bandes
        "mean": [0.0911, 0.0895, 0.0809, 0.0854, 0.1021, 0.2226, 0.2627,
                 0.2503, 0.2627, 0.2324, 0.1024, 0.0560, 0.0351],
        "std":  [0.0530, 0.0573, 0.0638, 0.0623, 0.0745, 0.1091, 0.1265,
                 0.1188, 0.1265, 0.1120, 0.0561, 0.0431, 0.0320],
    },
}


def get_norm_stats(num_bands: int) -> tuple[list[float], list[float]]:
    """Retourne (mean, std) pour num_bands bandes.
    Utilise des stats génériques si le nombre de bandes n'est pas tabulé."""
    if num_bands in BAND_NORM_STATS:
        s = BAND_NORM_STATS[num_bands]
        return s["mean"], s["std"]
    return [0.5] * num_bands, [0.25] * num_bands


def load_tiff(path: str, num_bands: int | None = None) -> np.ndarray:
    """Charge une image GeoTIFF et retourne un tableau float32 (H, W, C).

    Args:
        path: Chemin vers le fichier .tif.
        num_bands: Nombre de bandes à conserver (None = toutes).
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # (C, H, W)
        if num_bands is not None:
            if num_bands > img.shape[0]:
                raise ValueError(
                    f"{path} : l'image a {img.shape[0]} bandes, "
                    f"mais {num_bands} ont été demandées."
                )
            img = img[:num_bands]
        img = np.transpose(img, (1, 2, 0))  # → (H, W, C)
    return img


class SatelliteDataset(Dataset):
    """Dataset PyTorch pour images satellitaires multibandes."""

    def __init__(self, image_paths: list[str], labels: list[int], num_bands: int, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.num_bands = num_bands
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = load_tiff(self.image_paths[idx], self.num_bands)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class SatelliteClassifier(pl.LightningModule):
    """Classificateur ViT adapté aux images satellitaires multibandes.

    Remplace la couche Conv2d du patch embedding pour accepter num_bands
    canaux, en réutilisant les poids pré-entraînés RGB et en initialisant
    les bandes supplémentaires avec la moyenne des poids RGB.
    """

    def __init__(self, num_classes: int, num_bands: int = 3, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        if num_bands != 3:
            self._adapt_patch_embedding(num_bands)

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _adapt_patch_embedding(self, num_bands: int):
        """Remplace la projection du patch embedding pour num_bands canaux."""
        orig = self.vit.embeddings.patch_embeddings.projection  # Conv2d(3, D, 16, 16)
        new_proj = nn.Conv2d(
            num_bands,
            orig.out_channels,
            kernel_size=(orig.kernel_size[0], orig.kernel_size[1]),
            stride=(orig.stride[0], orig.stride[1]),
        )
        with torch.no_grad():
            if num_bands <= 3:
                new_proj.weight.data = orig.weight.data[:, :num_bands, :, :]
            else:
                # Copie les 3 premiers poids (RGB) et répète la moyenne pour les bandes sup.
                mean_w = orig.weight.data.mean(dim=1, keepdim=True)
                extra = mean_w.repeat(1, num_bands - 3, 1, 1)
                new_proj.weight.data = torch.cat([orig.weight.data, extra], dim=1)
            new_proj.bias.data = orig.bias.data.clone()
        self.vit.embeddings.patch_embeddings.projection = new_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


def create_dummy_data(num_bands: int = 4):
    """Génère un jeu de données factice (5 images × 2 classes × 2 splits)."""
    for split in ("train", "val"):
        for cls in ("class1", "class2"):
            os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)
            for i in range(5):
                data = np.random.randint(0, 256, (num_bands, 224, 224), dtype=np.uint8)
                path = f"dataset/{split}/{cls}/img_{i}.tif"
                with rasterio.open(
                    path, "w", driver="GTiff",
                    height=224, width=224, count=num_bands, dtype=data.dtype,
                ) as dst:
                    dst.write(data)


def main(num_bands: int = 4, num_classes: int = 2, max_epochs: int = 10, batch_size: int = 4):
    """Pipeline d'entraînement principal.

    Args:
        num_bands: Nombre de bandes spectrales (3=RGB, 4=RGB+NIR, 10/13=Sentinel-2).
        num_classes: Nombre de classes de sols à prédire.
        max_epochs: Nombre d'époques d'entraînement.
        batch_size: Taille du batch.
    """
    create_dummy_data(num_bands=num_bands)

    mean, std = get_norm_stats(num_bands)

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

    # Label déduit du nom du dossier parent (class1 → 0, class2 → 1, …)
    def label_from_path(p):
        return int(os.path.basename(os.path.dirname(p))[-1]) - 1

    train_labels = [label_from_path(p) for p in train_paths]
    val_labels   = [label_from_path(p) for p in val_paths]

    train_ds = SatelliteDataset(train_paths, train_labels, num_bands, transform=train_transform)
    val_ds   = SatelliteDataset(val_paths,   val_labels,   num_bands, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SatelliteClassifier(num_classes=num_classes, num_bands=num_bands)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
