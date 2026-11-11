"""Configuración central. Todo se puede sobrescribir con variables de entorno."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # backend/
PROJECT_DIR = BASE_DIR.parent                          # raíz del proyecto

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "ml_models" / "detector.pth"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", PROJECT_DIR / "frontend"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# Número de recortes de 224x224 que se promedian por imagen (1 = solo el central, máx. 5)
N_CROPS = int(os.getenv("N_CROPS", "5"))
