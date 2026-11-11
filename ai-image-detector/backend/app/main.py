"""API FastAPI: /api/health, /api/analyze y el frontend estático en /."""
import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_ORIGINS,
    FRONTEND_DIR,
    MAX_UPLOAD_MB,
    MODEL_PATH,
    N_CROPS,
)
from .ml.inference import Detector
from .ml.preprocessing import load_rgb

logger = logging.getLogger("detector")

# Protección contra "decompression bombs" (imágenes enormes que agotan la memoria)
Image.MAX_IMAGE_PIXELS = 60_000_000

app = FastAPI(title="Detector de imágenes creadas con IA", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector = Detector(MODEL_PATH, n_crops=N_CROPS)
if not detector.ready:
    logger.warning("No se encontró el modelo en %s. Entrena uno primero (ver README).", MODEL_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok", "max_upload_mb": MAX_UPLOAD_MB, **detector.info()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está entrenado. Ejecuta el entrenamiento (ver README) y reinicia el servidor.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Formato no soportado. Usa JPG, PNG o WEBP.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"La imagen supera el máximo de {MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img = load_rgb(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen; el archivo está dañado o no es una imagen.")

    result = await run_in_threadpool(detector.predict, img)
    result["filename"] = file.filename
    result["model_metrics"] = detector.metrics
    return result


# El frontend se sirve al final para que no tape las rutas /api
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
