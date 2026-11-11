"""Modelo para imágenes sueltas: ResNet-18 con una cabeza binaria.

Convención de etiquetas (la misma en todo el proyecto):
    0 = imagen REAL
    1 = imagen creada con IA
El modelo devuelve LOGITS; aplica sigmoid para obtener la probabilidad de IA.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .preprocessing import IMAGE_SIZE

CLASSES = ["real", "ai"]


class AIImageDetector(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.fc.in_features, 1))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> logits (B,)"""
        return self.backbone(x).squeeze(1)


def save_checkpoint(path, model: nn.Module, threshold: float, metrics: dict, extra: dict | None = None):
    """Guarda pesos + metadatos (umbral, métricas) solo con tipos simples, para poder
    cargarlo de forma segura con weights_only=True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "classes": CLASSES,
        "image_size": IMAGE_SIZE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = AIImageDetector(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
