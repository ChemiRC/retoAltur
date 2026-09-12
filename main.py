"""
main.py — Endpoint /detect para el reto Altur.

Endurecido contra los riesgos que te dejan en CERO el día del juez:
  - El documento fija el formato de RESPUESTA pero NO el nombre del campo del REQUEST.
    -> aceptamos varios nombres posibles Y base64 crudo en el body.
       (Aun así: CONFIRMA el esquema exacto con los ingenieros de Altur en sitio.)
  - Decodificación en memoria (sin archivos temporales) -> menor latencia.
  - Modelo cargado UNA vez al arranque.
  - Nunca revienta con 500 hacia el benchmark: si algo falla, responde JSON válido.

Arranque:
    uvicorn main:app --host 0.0.0.0 --port 8000
    (host 0.0.0.0, NO 127.0.0.1, para que el juez alcance tu endpoint;
     y expón con: ngrok http 8000  -> usa esa URL pública)
"""

import base64
import binascii
import json
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request

from features import extract_all

MODEL = None
FEATURE_NAMES = None
THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_NAMES, THRESHOLD
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            bundle = pickle.load(f)
        MODEL = bundle["model"]
        FEATURE_NAMES = bundle["feature_names"]
        print(f"Modelo cargado ({len(FEATURE_NAMES)} features).")
    else:
        print("[AVISO] model.pkl no encontrado -> modo degradado (siempre responde JSON válido).")
    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            content = f.read().lstrip("\ufeff")  # quita BOM si existe
            THRESHOLD = json.loads(content).get("threshold", 0.5)
        print(f"Umbral cargado: {THRESHOLD:.4f}")
    yield


app = FastAPI(title="Altur Voice Anti-Spoofing", lifespan=lifespan)

# Campos donde podría venir el base64. Ajusta si Altur usa otro nombre.
_CANDIDATE_FIELDS = ["audio_base64", "audio", "wav_base64", "wav", "data", "clip", "file"]


async def _extract_b64(request: Request):
    """Saca el base64 del request sin importar cómo lo empaqueten los jueces."""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        if isinstance(body, str):
            return body
        for k in _CANDIDATE_FIELDS:
            if isinstance(body, dict) and k in body and body[k]:
                return body[k]
        # último recurso: primer valor string largo del dict
        if isinstance(body, dict):
            for v in body.values():
                if isinstance(v, str) and len(v) > 100:
                    return v
        return None
    # body crudo (texto base64 o bytes)
    raw = await request.body()
    try:
        return raw.decode("utf-8").strip().strip('"')
    except Exception:
        return raw  # ya son bytes de audio


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "threshold": THRESHOLD}


@app.post("/detect")
async def detect(request: Request):
    try:
        b64 = await _extract_b64(request)
        if b64 is None:
            return {"is_synthetic": False, "confidence": 0.5, "error": "no_audio_field"}

        # Decodificar a bytes de audio
        if isinstance(b64, (bytes, bytearray)):
            audio_bytes = bytes(b64)
        else:
            try:
                audio_bytes = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError):
                return {"is_synthetic": False, "confidence": 0.5, "error": "bad_base64"}

        vec, names = extract_all(audio_bytes)

        # Alinear features con las del entrenamiento (por si cambia el orden)
        if MODEL is not None and FEATURE_NAMES is not None:
            index = {n: i for i, n in enumerate(names)}
            row = np.array([vec[index[n]] if n in index else 0.0
                            for n in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
            prob_synth = float(MODEL.predict_proba(row)[0, 1])
        else:
            # Modo degradado sin modelo: responde neutral (no rompe el benchmark)
            prob_synth = 0.5

        return {
            "is_synthetic": bool(prob_synth >= THRESHOLD),
            "confidence": round(prob_synth, 4),
        }

    except Exception as e:
        # NUNCA devolver 500 al benchmark: mejor un veredicto neutral que un crash.
        return {"is_synthetic": False, "confidence": 0.5, "error": str(e)[:200]}