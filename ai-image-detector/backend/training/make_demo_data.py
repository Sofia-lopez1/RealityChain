"""Genera un dataset SINTÉTICO diminuto solo para comprobar que el proyecto corre de punta a punta.

No sirve para detectar IA real: las imágenes son campos de color suaves; las "fake" llevan
una trama periódica de alta frecuencia. Para un detector de verdad usa un dataset real (ver README).

    python -m training.make_demo_data --out ../demo_data --n 80
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def smooth_field(rng, size=256):
    small = (rng.random((8, 8, 3)) * 255).astype("uint8")
    arr = np.asarray(Image.fromarray(small).resize((size, size), Image.BICUBIC)).astype(np.float32)
    return arr + rng.normal(0, 6, arr.shape)


def add_grid(arr, period=4, amp=14):
    yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]]
    pattern = np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period) * amp
    return arr + pattern[..., None]


def save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(path, format="PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../demo_data")
    ap.add_argument("--n", type=int, default=80, help="Imágenes por clase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for name in ("real", "fake"):
        (out / name).mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        save(smooth_field(rng), out / "real" / f"real_{i:04d}.png")
        save(add_grid(smooth_field(rng)), out / "fake" / f"fake_{i:04d}.png")
    print(f"Dataset sintético creado en {out.resolve()} ({args.n} por clase)")


if __name__ == "__main__":
    main()
