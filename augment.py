"""
augment.py - Aumentacion de audio realista para telefonia.

Objetivo: que el modelo NO se aferre a un artefacto de codec/pipeline especifico
(el atajo 'zcr_std' que da 100% en limpio pero 38% EER bajo codec). Simulamos
condiciones de linea variadas -> robustez ante 'unseen call conditions' del reto.

OJO sobre alcance: esto endurece contra CANALES nuevos (codec, ruido, ganancia),
NO contra ENGINES de TTS nuevos. El hedge contra engines nuevos son las features
de timing conversacional (un TTS nuevo sigue teniendo turnos roboticamente parejos).
"""
import numpy as np


def mulaw_roundtrip(x, mu=255):
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y * 0.5 + 0.5) * 255)
    yq = (q / 255.0 - 0.5) * 2
    xr = np.sign(yq) * (1 / mu) * ((1 + mu) ** np.abs(yq) - 1)
    return xr.astype(np.float32)


def augment_stereo(caller, agent, rng):
    """Devuelve (caller', agent') con una degradacion telefonica aleatoria."""
    c, a = caller.copy(), agent.copy()
    mode = rng.integers(0, 4)
    if mode == 0:   # codec G.711 mu-law
        c, a = mulaw_roundtrip(c), mulaw_roundtrip(a)
    elif mode == 1: # ruido de linea
        c = c + rng.normal(0, rng.uniform(0.001, 0.004), len(c)).astype(np.float32)
        a = a + rng.normal(0, rng.uniform(0.001, 0.004), len(a)).astype(np.float32)
    elif mode == 2: # variacion de ganancia + clip suave
        g = rng.uniform(0.6, 1.4)
        c = np.clip(c * g, -1, 1); a = np.clip(a * g, -1, 1)
    else:           # codec + ruido juntos (lo mas duro)
        c = mulaw_roundtrip(c) + rng.normal(0, 0.002, len(c)).astype(np.float32)
        a = mulaw_roundtrip(a) + rng.normal(0, 0.002, len(a)).astype(np.float32)
    return c.astype(np.float32), a.astype(np.float32)