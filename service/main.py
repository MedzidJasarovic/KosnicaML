"""
FastAPI servis za prepoznavanje zdravstvenog stanja pčela sa fotografije.

    GET  /health   — da li servis radi i koji model je učitan
    POST /predict  — multipart polje "file" (slika) -> {label, confidence, all_probs, model_version}

Pokretanje (iz KosnicaML/):
    ./.venv/bin/uvicorn service.main:app --port 8000 --reload

Putanja do modela može se promeniti promenljivom okruženja MODEL_PATH.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .inference import BeeHealthModel

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "bee_health.onnx"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

model: BeeHealthModel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Model se učitava jednom, pri pokretanju — ne pri svakom zahtevu.
    global model
    model = BeeHealthModel(MODEL_PATH)
    yield


app = FastAPI(title="Košnica — zdravlje pčela", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model.version,
        "arch": model.arch,
        "classes": model.classes,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Prazan fajl.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Slika je veća od 10 MB.")
    try:
        return model.predict(data)
    except UnidentifiedImageError:
        raise HTTPException(400, "Fajl nije slika ili format nije podržan.")
