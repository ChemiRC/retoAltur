"""
features.py — Extracción de características para el reto Altur.

DOS familias de señales:
  1. Acústicas (canal 0): MFCC + deltas + estadísticos espectrales + chroma + contrast
  2. Timing conversacional: latencias, CV, solapamientos, pausas
     - Puede usar turns/ JSON oficiales (entrenamiento) o VAD de energía (producción)

REGLA: extract_all(audio_bytes) = producción (VAD). Mismo código en train y serve.
       extract_all_with_turns(audio_bytes, turns_list) = entrenamiento mejorado.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 8000
N_MFCC = 20
FRAME_MS = 25
HOP_MS = 10
MAX_RESPONSE_GAP = 3.0
MERGE_GAP = 0.35
MIN_SEG = 0.20


def load_stereo_from_bytes(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    data = data.T
    if data.shape[0] == 1:
        caller = data[0]
        agent = np.zeros_like(caller)
    else:
        caller = data[0]
        agent = data[1]
    if sr != TARGET_SR:
        caller = librosa.resample(caller, orig_sr=sr, target_sr=TARGET_SR)
        agent = librosa.resample(agent, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return caller, agent, sr


def _merge_segments(segs, gap=MERGE_GAP, min_dur=MIN_SEG):
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) >= min_dur]


def energy_vad(y, sr):
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(y) < frame:
        return []
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    if rms.max() <= 0:
        return []
    thr = max(np.median(rms) * 1.5, rms.max() * 0.06)
    active = rms > thr
    segments = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg, start = True, i
        elif not a and in_seg:
            in_seg = False
            segments.append((start * hop / sr, i * hop / sr))
    if in_seg:
        segments.append((start * hop / sr, len(active) * hop / sr))
    return _merge_segments(segments)


def _safe_stats(arr, prefix):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0, f"{prefix}_max": 0.0, f"{prefix}_n": 0.0}
    return {f"{prefix}_mean": float(np.mean(a)), f"{prefix}_std": float(np.std(a)),
            f"{prefix}_min": float(np.min(a)), f"{prefix}_max": float(np.max(a)),
            f"{prefix}_n": float(a.size)}


# ---------------------------------------------------------------------------
# 1) Features ACÚSTICAS — ahora con chroma y spectral contrast
# ---------------------------------------------------------------------------
def acoustic_features(caller, sr):
    feats = {}

    # MFCC + deltas (señal temporal del tracto vocal)
    mfcc = librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    for name, m in [("mfcc", mfcc), ("d1", d1), ("d2", d2)]:
        for i in range(m.shape[0]):
            feats[f"{name}_{i}_mean"] = float(np.mean(m[i]))
            feats[f"{name}_{i}_std"] = float(np.std(m[i]))

    # Estadísticos espectrales básicos
    centroid  = librosa.feature.spectral_centroid(y=caller, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=caller, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=caller, sr=sr)[0]
    flatness  = librosa.feature.spectral_flatness(y=caller)[0]
    zcr       = librosa.feature.zero_crossing_rate(y=caller)[0]
    rmse      = librosa.feature.rms(y=caller)[0]
    for name, v in [("centroid", centroid), ("bandwidth", bandwidth),
                    ("rolloff", rolloff), ("flatness", flatness),
                    ("zcr", zcr), ("rmse", rmse)]:
        feats[f"{name}_mean"] = float(np.mean(v))
        feats[f"{name}_std"]  = float(np.std(v))

    # NUEVO: Chroma — captura contenido armónico (TTS más artificialmente "correcto")
    chroma = librosa.feature.chroma_stft(y=caller, sr=sr)
    feats["chroma_mean"] = float(np.mean(chroma))
    feats["chroma_std"]  = float(np.std(chroma))
    feats["chroma_max"]  = float(np.max(chroma))

    # NUEVO: Spectral contrast — diferencia entre picos y valles espectrales por banda
    # TTS tiende a tener contraste más uniforme/artificial
    try:
        contrast = librosa.feature.spectral_contrast(y=caller, sr=sr, n_bands=4)
        for i in range(contrast.shape[0]):
            feats[f"contrast_{i}_mean"] = float(np.mean(contrast[i]))
            feats[f"contrast_{i}_std"]  = float(np.std(contrast[i]))
    except Exception:
        for i in range(5):
            feats[f"contrast_{i}_mean"] = 0.0
            feats[f"contrast_{i}_std"]  = 0.0

    return feats


# ---------------------------------------------------------------------------
# 2) Features de TIMING — dos versiones: VAD (producción) y turns JSON (entrenamiento)
# ---------------------------------------------------------------------------
def _timing_from_segments(caller_seg, agent_seg, duration):
    """Núcleo compartido: recibe segmentos y calcula todas las features de timing."""
    feats = {}
    dur = max(duration, 1e-6)
    caller_talk = sum(e - s for s, e in caller_seg)
    agent_talk  = sum(e - s for s, e in agent_seg)
    feats["caller_talk_ratio"] = caller_talk / dur
    feats["agent_talk_ratio"]  = agent_talk / dur
    feats["n_caller_turns"]    = float(len(caller_seg))
    feats["n_agent_turns"]     = float(len(agent_seg))

    # Latencia de respuesta del llamante tras cada turno del agente
    caller_starts = sorted(s for s, e in caller_seg)
    latencies = []
    for _, a_end in agent_seg:
        nxt = [cs for cs in caller_starts if cs >= a_end]
        if nxt:
            gap = nxt[0] - a_end
            if 0 <= gap <= MAX_RESPONSE_GAP:
                latencies.append(gap)
    feats.update(_safe_stats(latencies, "resp_latency"))
    if latencies and np.mean(latencies) > 0:
        feats["resp_latency_cv"] = float(np.std(latencies) / np.mean(latencies))
    else:
        feats["resp_latency_cv"] = 0.0

    # NUEVO: duración de turnos del llamante (varianza = naturalidad)
    caller_durs = [e - s for s, e in caller_seg]
    feats.update(_safe_stats(caller_durs, "caller_turn_dur"))

    # NUEVO: duración de turnos del agente
    agent_durs = [e - s for s, e in agent_seg]
    feats.update(_safe_stats(agent_durs, "agent_turn_dur"))

    # Solapamientos / interrupciones
    overlap = 0.0
    for cs, ce in caller_seg:
        for as_, ae in agent_seg:
            overlap += max(0.0, min(ce, ae) - max(cs, as_))
    feats["overlap_ratio"] = overlap / dur

    # Pausas internas del llamante
    pauses = []
    cs_sorted = sorted(caller_seg)
    for i in range(1, len(cs_sorted)):
        pauses.append(cs_sorted[i][0] - cs_sorted[i - 1][1])
    feats.update(_safe_stats(pauses, "caller_pause"))

    return feats


def timing_features(caller, agent, sr):
    """Versión VAD — usada en PRODUCCIÓN (sin turns JSON)."""
    caller_seg = energy_vad(caller, sr)
    agent_seg  = energy_vad(agent, sr)
    duration   = max(len(caller) / sr, 1e-6)
    return _timing_from_segments(caller_seg, agent_seg, duration)


def timing_features_from_turns(turns_list, duration):
    """Versión turns JSON — usada en ENTRENAMIENTO (más limpia que VAD)."""
    caller_seg = sorted((t["start"], t["end"]) for t in turns_list if t["channel"] == 0)
    agent_seg  = sorted((t["start"], t["end"]) for t in turns_list if t["channel"] == 1)
    return _timing_from_segments(caller_seg, agent_seg, duration)


# ---------------------------------------------------------------------------
# Orquestadores
# ---------------------------------------------------------------------------
def extract_from_arrays(caller, agent, sr, turns_list=None):
    feats = {}
    feats.update(acoustic_features(caller, sr))
    if turns_list is not None:
        duration = max(len(caller) / sr, 1e-6)
        feats.update(timing_features_from_turns(turns_list, duration))
    else:
        feats.update(timing_features(caller, agent, sr))
    names = sorted(feats.keys())
    vec = np.array([feats[k] for k in names], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names


def extract_all(audio_bytes):
    """Producción: solo el WAV, sin turns JSON."""
    caller, agent, sr = load_stereo_from_bytes(audio_bytes)
    return extract_from_arrays(caller, agent, sr, turns_list=None)


if __name__ == "__main__":
    sr = 8000
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    caller = 0.1 * np.sin(2 * np.pi * 180 * t) * (np.random.rand(len(t)) > 0.3)
    agent  = 0.1 * np.sin(2 * np.pi * 120 * t) * (np.random.rand(len(t)) > 0.6)
    stereo = np.stack([caller, agent], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    vec, names = extract_all(buf.getvalue())
    print("Nº de features:", len(vec))
    print("Nuevas features:", [n for n in names if "chroma" in n or "contrast" in n or "turn_dur" in n])