"""Evalúa un checkpoint sobre el conjunto de test (datos que nunca vio al entrenar).

    python -m training.evaluate --data ../data
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config import BASE_DIR
from app.ml.model import load_checkpoint

from .dataset import build_datasets
from .metrics import compute_metrics
from .train import predict_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", default=str(BASE_DIR / "ml_models" / "detector.pth"))
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42, help="Debe coincidir con la semilla usada al entrenar")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    threshold = ckpt.get("threshold", 0.5)

    _, val_ds, test_ds = build_datasets(args.data, seed=args.seed, max_per_class=args.max_per_class)
    ds = test_ds if args.split == "test" else val_ds
    if ds is None:
        raise SystemExit("Este dataset no tiene conjunto de test.")

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    _, probs, labels = predict_loader(model, loader, nn.BCEWithLogitsLoss(), device)
    m = compute_metrics(probs, labels, threshold)

    print(f"Conjunto: {args.split} ({len(ds)} imágenes) | umbral: {threshold:.2f}")
    print(f"Accuracy:              {m['accuracy'] * 100:.2f}%")
    print(f"Accuracy balanceada:   {m['balanced_accuracy'] * 100:.2f}%")
    print(f"AUC:                   {m['auc']:.4f}")
    print(f"Reales bien clasif.:   {m['recall_real'] * 100:.2f}%")
    print(f"IA bien clasificadas:  {m['recall_ai'] * 100:.2f}%")
    print(f"Precisión (IA):        {m['precision_ai'] * 100:.2f}% | F1: {m['f1_ai']:.3f}")
    print("\nMatriz de confusión (filas = verdad, columnas = predicción)")
    print(f"              pred real   pred IA")
    print(f"  real        {m['tn']:>9}  {m['fp']:>8}")
    print(f"  IA          {m['fn']:>9}  {m['tp']:>8}")


if __name__ == "__main__":
    main()
