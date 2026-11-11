"""Entrenamiento del detector de imágenes IA vs reales.

Ejecutar desde la carpeta backend/:
    python -m training.train --data ../data --epochs 10

Cada época se evalúa sobre VALIDACIÓN (datos que el modelo no usa para aprender),
se guarda el mejor checkpoint según el AUC de validación y se calcula el umbral
de decisión óptimo. Una accuracy de entrenamiento alta NO significa que funcione:
mira siempre las métricas de validación.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config import BASE_DIR
from app.ml.model import AIImageDetector, save_checkpoint

from .dataset import build_datasets
from .metrics import best_threshold, compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model, loader, criterion, device):
    model.eval()
    probs, labels, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        probs.append(torch.sigmoid(logits).float().cpu())
        labels.append(y.cpu())
    return total_loss / max(n, 1), torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Carpeta del dataset (ver README)")
    ap.add_argument("--out", default=str(BASE_DIR / "ml_models" / "detector.pth"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4, help="Épocas sin mejorar antes de parar")
    ap.add_argument("--max-per-class", type=int, default=None, help="Limita imágenes por clase (pruebas rápidas)")
    ap.add_argument("--no-pretrained", action="store_true", help="No descargar pesos de ImageNet (solo para pruebas)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_ds, val_ds, test_ds = build_datasets(args.data, seed=args.seed, max_per_class=args.max_per_class)
    n_real = sum(1 for _, l in train_ds.samples if l == 0)
    n_fake = len(train_ds) - n_real
    print(f"Train: {len(train_ds)} (real={n_real}, IA={n_fake}) | Val: {len(val_ds)} | "
          f"Test: {len(test_ds) if test_ds else 'no hay'}")
    if n_real == 0 or n_fake == 0:
        raise SystemExit("Hace falta al menos una imagen de cada clase en train.")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = AIImageDetector(pretrained=not args.no_pretrained).to(device)
    pos_weight = torch.tensor([n_real / n_fake], device=device)  # compensa clases desbalanceadas
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_auc, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        val_loss, val_probs, val_labels = predict_loader(model, val_loader, eval_criterion, device)
        threshold = best_threshold(val_probs, val_labels)
        m = compute_metrics(val_probs, val_labels, threshold)
        print(f"Época {epoch:>2}/{args.epochs} | train_loss {running / seen:.4f} | val_loss {val_loss:.4f} | "
              f"val_acc {m['accuracy'] * 100:.1f}% | AUC {m['auc']:.3f} | "
              f"reales OK {m['recall_real'] * 100:.1f}% | IA OK {m['recall_ai'] * 100:.1f}% | "
              f"umbral {threshold:.2f} | {time.time() - t0:.0f}s")

        score = m["auc"] if not np.isnan(m["auc"]) else m["balanced_accuracy"]
        if score > best_auc:
            best_auc, best_epoch, stale = score, epoch, 0
            save_checkpoint(args.out, model, threshold, m, extra={"epoch": epoch})
            print(f"   ✔ Mejor modelo hasta ahora guardado en {args.out}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Sin mejora en {args.patience} épocas: parada anticipada.")
                break

    print(f"\nListo. Mejor época: {best_epoch} (AUC val = {best_auc:.3f}).")
    print("Evalúa con datos nunca vistos:  python -m training.evaluate --data <carpeta>")


if __name__ == "__main__":
    main()
