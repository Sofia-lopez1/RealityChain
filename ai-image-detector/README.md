# Detector de imágenes creadas con IA

Proyecto completo (backend + frontend + entrenamiento) para clasificar una imagen como **real** o **creada con IA**.

```
ai-image-detector/
├── backend/
│   ├── app/
│   │   ├── main.py              # API FastAPI (/api/health, /api/analyze) y sirve el frontend
│   │   ├── config.py            # configuración (variables de entorno)
│   │   └── ml/
│   │       ├── model.py         # ResNet-18 + cabeza binaria, guardado/carga de checkpoints
│   │       ├── preprocessing.py # preprocesado COMPARTIDO por entrenamiento e inferencia
│   │       └── inference.py     # servicio de predicción (multi-recorte)
│   ├── training/
│   │   ├── dataset.py           # carga real/fake con split train/val/test
│   │   ├── train.py             # entrenamiento con validación y mejor checkpoint
│   │   ├── evaluate.py          # evaluación final sobre test
│   │   ├── metrics.py
│   │   └── make_demo_data.py    # datos sintéticos SOLO para probar el pipeline
│   ├── ml_models/               # aquí se guarda detector.pth
│   └── requirements.txt
├── frontend/                    # HTML + CSS + JS puro (sin build)
└── data/{real,fake}/            # tu dataset
```

## Qué cambió respecto a tu proyecto original

| Problema | Solución |
|---|---|
| El dataset era ruido aleatorio con etiquetas al azar | Dataset real con carpetas `real/` y `fake/` |
| CNN+LSTM sobre una imagen repetida 10 veces (sin información temporal) | Una CNN (ResNet-18 preentrenada) para imágenes sueltas |
| Accuracy medida solo en entrenamiento | Split train/val/test; el mejor modelo se elige por AUC de validación |
| `Resize` a 224 destruye los rastros de la IA | Sin reducción: recortes de tamaño nativo (aleatorios al entrenar, 5 al predecir) |
| `BCELoss` + `Sigmoid` dentro del modelo | `BCEWithLogitsLoss` (más estable); el sigmoid se aplica solo al predecir |
| Etiquetas ambiguas (¿1 = real o IA?) | Convención única: **0 = real, 1 = IA** |
| Umbral fijo de 0.5 | Umbral óptimo calculado en validación y guardado en el checkpoint |
| Riesgo de preprocesar distinto al predecir | Mismo módulo `preprocessing.py` en todo |

## 1. Instalación

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Con GPU NVIDIA instala PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/ (el entrenamiento en CPU funciona pero es lento).

## 2. Dataset

Es **lo más importante**: el modelo solo aprende de lo que le des. Estructura soportada (mayúsculas/minúsculas da igual):

```
data/                       data/
├── real/*.jpg              ├── train/{REAL,FAKE}/...
└── fake/*.jpg              ├── valid/{REAL,FAKE}/...   (opcional)
   (se divide solo)         └── test/{REAL,FAKE}/...    (opcional)
```

Datasets públicos recomendados:

- **CIFAKE** (Kaggle): 60k reales + 60k IA, 32×32. Bueno para probar rápido, pero imágenes diminutas.
- **140k Real and Fake Faces** (Kaggle): caras reales vs. StyleGAN.
- **GenImage**, **ArtiFact**, **Diffusion-Forensics**: imágenes de generadores modernos (Stable Diffusion, Midjourney, DALL·E…). Los mejores para uso real.
- Para video: FaceForensics++, Celeb-DF, DFDC (requeriría extraer frames; ver más abajo).

Con la CLI de Kaggle: `kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images`.

**Reglas para que generalice:**
1. Mezcla **varios generadores** de IA. Un modelo entrenado con uno solo suele fallar con los demás.
2. Evita sesgos: si todas las reales son JPG y todas las IA son PNG (o tienen otra resolución), el modelo aprenderá eso y no la IA. Procura mismo formato y resoluciones parecidas en ambas clases.
3. Miles de imágenes por clase como mínimo (idealmente decenas de miles).

## 3. Probar que todo funciona (opcional, 2 minutos)

```bash
python -m training.make_demo_data --out ../demo_data --n 80
python -m training.train --data ../demo_data --epochs 3 --workers 0
```

Esos datos son sintéticos: sirven solo para comprobar que el flujo corre. No detectan IA real.

## 4. Entrenar con tu dataset

```bash
cd backend
python -m training.train --data ../data --epochs 10 --batch-size 32
python -m training.evaluate --data ../data          # métricas sobre test
```

El entrenamiento imprime por época: pérdida, accuracy de validación, AUC, **% de reales bien clasificadas** y **% de IA bien clasificadas** (si una de las dos es muy baja, el modelo está sesgado hacia una clase). Guarda `backend/ml_models/detector.pth` con los pesos, el umbral óptimo y las métricas.

Opciones útiles: `--lr 3e-4`, `--epochs 20`, `--max-per-class 5000` (pruebas rápidas), `--workers 8`.

## 5. Ejecutar la aplicación

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Abre **http://localhost:8000** (el backend sirve el frontend). Documentación interactiva de la API en `/docs`.

Si sirves el frontend aparte (p. ej. Live Server), edita `frontend/config.js` y pon `API_BASE: "http://localhost:8000"`.

### API

`POST /api/analyze` (multipart, campo `file`) →

```json
{
  "label": "ai",
  "label_es": "Creada con IA",
  "probability_ai": 0.9132,
  "probability_real": 0.0868,
  "certainty": "alta",
  "threshold": 0.47,
  "crops_analyzed": 5,
  "crop_disagreement": 0.04,
  "inference_ms": 180.2,
  "model_metrics": { "accuracy": 0.94, "auc": 0.98, "...": "..." }
}
```

Errores: `415` formato no soportado, `413` demasiado grande, `400` archivo ilegible, `503` modelo sin entrenar.

Variables de entorno: `MODEL_PATH`, `MAX_UPLOAD_MB` (10), `N_CROPS` (5), `ALLOWED_ORIGINS` (`*`, restríngelo en producción), `FRONTEND_DIR`.

## 6. Si sigue sin acertar

1. Mira las métricas de **validación/test**, no las de entrenamiento. Si la accuracy de train sube y la de validación no, hay sobreajuste: más datos, más variedad o más augmentations.
2. Si clasifica todo como una sola clase: dataset desbalanceado o con sesgo (formato/resolución distintos entre clases).
3. Si funciona en tu test pero falla con imágenes que subes tú: tus imágenes vienen de un generador que no estaba en el entrenamiento (o están muy comprimidas por redes sociales). Añade ejemplos de ese tipo.
4. Prueba un backbone mayor (ResNet-50, EfficientNet) cambiando `model.py`; si lo haces, reentrena.

## Límites honestos

Detectar imágenes generadas por IA es un problema abierto. Ningún detector es infalible: los generadores mejoran constantemente, y la compresión, el recorte o la edición borran rastros. Trata el resultado como un indicio, no como una prueba, y no lo uses como única base para decisiones importantes.

## ¿Y si trabajas con video?

La arquitectura CNN+LSTM original solo tiene sentido con frames **consecutivos reales** de un video. Para ese caso: extrae ~10 frames por video, pásalos por esta misma CNN y promedia las probabilidades (lo más simple y robusto), o añade después una LSTM sobre los vectores de características.
