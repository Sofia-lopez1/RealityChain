"""Preprocesado compartido por entrenamiento, evaluación e inferencia.

Idea clave: los rastros de una imagen generada por IA viven en los detalles finos
de los píxeles. Por eso NUNCA reducimos la imagen con Resize: solo la ampliamos si
es más pequeña que el recorte, y trabajamos con recortes de tamaño nativo.
"""
import io
import random

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def load_rgb(source) -> Image.Image:
    """Abre una imagen (ruta o archivo en memoria), corrige la orientación EXIF y la pasa a RGB."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class EnsureMinSize:
    """Amplía la imagen solo si su lado corto es menor que `size`. Nunca la reduce."""

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        short = min(w, h)
        if short >= self.size:
            return img
        scale = self.size / short
        new_w = max(self.size, round(w * scale))
        new_h = max(self.size, round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)


class RandomJPEG:
    """Recompresión JPEG aleatoria, para que el modelo no dependa del formato del archivo."""

    def __init__(self, p: float = 0.3, quality=(60, 95)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    """Desenfoque gaussiano leve ocasional."""

    def __init__(self, p: float = 0.1, radius=(0.2, 1.0)):
        self.p = p
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


_to_normalized_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)


def build_train_transform(size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            EnsureMinSize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            RandomJPEG(p=0.3),
            RandomBlur(p=0.1),
            _to_normalized_tensor,
        ]
    )


def build_eval_transform(size: int = IMAGE_SIZE):
    return transforms.Compose(
        [EnsureMinSize(size), transforms.CenterCrop(size), _to_normalized_tensor]
    )


def _crop_positions(w: int, h: int, size: int, n: int):
    """Centro + 4 esquinas, sin repetir posiciones (imágenes pequeñas generan menos recortes)."""
    candidates = [
        ((w - size) // 2, (h - size) // 2),
        (0, 0),
        (w - size, 0),
        (0, h - size),
        (w - size, h - size),
    ]
    out = []
    for pos in candidates[: max(1, min(n, 5))]:
        if pos not in out:
            out.append(pos)
    return out


def multi_crop_tensor(img: Image.Image, size: int = IMAGE_SIZE, n_crops: int = 5) -> torch.Tensor:
    """Devuelve un tensor (k, 3, size, size) con k recortes de la imagen para inferencia."""
    img = EnsureMinSize(size)(img)
    w, h = img.size
    crops = [
        _to_normalized_tensor(img.crop((x, y, x + size, y + size)))
        for x, y in _crop_positions(w, h, size, n_crops)
    ]
    return torch.stack(crops)
