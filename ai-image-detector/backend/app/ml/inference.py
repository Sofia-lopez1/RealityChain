"""Servicio de inferencia: carga el checkpoint una vez y predice sobre imágenes PIL."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from PIL import Image

from .model import load_checkpoint
from .preprocessing import IMAGE_SIZE, multi_crop_tensor


def _clean(metrics: dict) -> dict:
    """NaN/inf no son JSON válido; los convertimos en None."""
    return {k: (v if isinstance(v, (int, float)) and math.isfinite(v) else None) for k, v in metrics.items()}


def _certainty(p_ai: float, threshold: float) -> str:
    """Qué tan lejos está la probabilidad del umbral de decisión (normalizado a 0..1)."""
    if p_ai >= threshold:
        d = (p_ai - threshold) / max(1e-9, 1 - threshold)
    else:
        d = (threshold - p_ai) / max(1e-9, threshold)
    return "alta" if d >= 0.66 else "media" if d >= 0.33 else "baja"


class Detector:
    def __init__(self, model_path, n_crops: int = 5, device: str | None = None):
        self.model_path = Path(model_path)
        self.n_crops = n_crops
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.threshold = 0.5
        self.metrics: dict = {}
        self.created_at: str | None = None
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            self.model = None
            return
        self.model, ckpt = load_checkpoint(self.model_path, self.device)
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.metrics = _clean(ckpt.get("metrics", {}))
        self.created_at = ckpt.get("created_at")

    def info(self) -> dict:
        return {
            "model_loaded": self.ready,
            "device": str(self.device),
            "threshold": self.threshold if self.ready else None,
            "metrics": self.metrics if self.ready else {},
            "trained_at": self.created_at,
        }

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> dict:
        if not self.ready:
            raise RuntimeError("El modelo no está cargado.")
        t0 = time.perf_counter()
        batch = multi_crop_tensor(img, IMAGE_SIZE, self.n_crops).to(self.device)
        probs = torch.sigmoid(self.model(batch)).float().cpu()
        p_ai = float(probs.mean())
        is_ai = p_ai >= self.threshold
        return {
            "label": "ai" if is_ai else "real",
            "label_es": "Creada con IA" if is_ai else "Real",
            "probability_ai": round(p_ai, 4),
            "probability_real": round(1 - p_ai, 4),
            "certainty": _certainty(p_ai, self.threshold),
            "threshold": round(self.threshold, 4),
            "crops_analyzed": int(probs.numel()),
            "crop_disagreement": round(float(probs.std(unbiased=False)), 4),
            "image_size": list(img.size),
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
