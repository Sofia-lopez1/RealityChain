"""Carga del dataset real/fake.

Estructuras soportadas (los nombres de carpeta no distinguen mayúsculas):

  A) Plana: se divide automáticamente en train/val/test (estratificado, con semilla)
       data/real/*.jpg
       data/fake/*.jpg

  B) Ya dividida (CIFAKE, 140k Real and Fake Faces, etc.). `val` puede llamarse
     val / valid / validation, y `test` y `val` son opcionales:
       data/train/{REAL,FAKE}/...
       data/valid/{REAL,FAKE}/...
       data/test/{REAL,FAKE}/...

Etiquetas: real = 0, fake (IA) = 1.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from app.ml.preprocessing import build_eval_transform, build_train_transform, load_rgb

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def _child_dir(base: Path, names) -> Path | None:
    if not base.is_dir():
        return None
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def _scan(base: Path, max_per_class: int | None = None, seed: int = 42):
    real_dir = _child_dir(base, ("real",))
    fake_dir = _child_dir(base, ("fake", "ai"))
    if real_dir is None or fake_dir is None:
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'real' y 'fake' dentro de {base}. "
            "Revisa la estructura descrita en el README."
        )
    samples = []
    for label, folder in ((0, real_dir), (1, fake_dir)):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
        if max_per_class and len(files) > max_per_class:
            files = random.Random(seed).sample(files, max_per_class)
        samples += [(str(p), label) for p in files]
    return samples


def _stratified_split(samples, frac: float, seed: int):
    """Separa `frac` de cada clase. Devuelve (resto, separado)."""
    rng = random.Random(seed)
    rest, taken = [], []
    for label in (0, 1):
        group = [s for s in samples if s[1] == label]
        rng.shuffle(group)
        k = int(len(group) * frac)
        taken += group[:k]
        rest += group[k:]
    return rest, taken


class BinaryImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(5):  # si una imagen está corrupta, probamos con otra
            path, label = self.samples[idx]
            try:
                img = load_rgb(path)
                break
            except Exception:
                idx = random.randrange(len(self.samples))
        else:
            raise RuntimeError("Demasiadas imágenes ilegibles seguidas; revisa el dataset.")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def build_datasets(root, val_frac=0.15, test_frac=0.15, seed=42, max_per_class=None):
    """Devuelve (train_ds, val_ds, test_ds). test_ds puede ser None si no existe."""
    root = Path(root)
    dirs = {s: _child_dir(root, names) for s, names in SPLIT_ALIASES.items()}

    if dirs["train"] is not None:
        train = _scan(dirs["train"], max_per_class, seed)
        val = _scan(dirs["val"], max_per_class, seed) if dirs["val"] else None
        test = _scan(dirs["test"], max_per_class, seed) if dirs["test"] else None
        if val is None:
            train, val = _stratified_split(train, val_frac, seed)
    else:
        flat = _scan(root, max_per_class, seed)
        rest, test = _stratified_split(flat, test_frac, seed)
        train, val = _stratified_split(rest, val_frac / (1 - test_frac), seed + 1)

    if not train or not val:
        raise RuntimeError("El dataset es demasiado pequeño para dividirlo en train/val.")

    train_ds = BinaryImageDataset(train, build_train_transform())
    val_ds = BinaryImageDataset(val, build_eval_transform())
    test_ds = BinaryImageDataset(test, build_eval_transform()) if test else None
    return train_ds, val_ds, test_ds
