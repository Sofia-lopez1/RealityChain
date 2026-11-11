import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(probs, labels, threshold: float = 0.5) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall_ai = tp / max(tp + fn, 1)       # de las imágenes IA, cuántas detecta
    recall_real = tn / max(tn + fp, 1)     # de las reales, cuántas reconoce como reales
    f1 = 2 * precision * recall_ai / max(precision + recall_ai, 1e-9)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:  # solo hay una clase en los datos
        auc = float("nan")

    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": (recall_ai + recall_real) / 2,
        "precision_ai": precision,
        "recall_ai": recall_ai,
        "recall_real": recall_real,
        "f1_ai": f1,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def best_threshold(probs, labels) -> float:
    """Umbral que maximiza la exactitud balanceada (reales e IA pesan igual)."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        score = compute_metrics(probs, labels, float(t))["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_t, best_score = float(t), score
    return best_t
