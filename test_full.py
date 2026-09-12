import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64, requests
import pandas as pd
import numpy as np

URL = "http://127.0.0.1:8000/detect"
df = pd.read_csv("manifest.csv")
val = df[df["split"] == "val"].reset_index(drop=True)

print(f"Probando {len(val)} audios del set de validación...\n")

resultados = []
for i, row in val.iterrows():
    path = os.path.join("audio", row["anon_id"] + ".wav")
    if not os.path.exists(path):
        print(f"  [FALTA] {row['anon_id']}")
        continue
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    resp = requests.post(URL, json={"audio_base64": b64}).json()
    resultados.append({
        "id": row["anon_id"],
        "real": row["label"],
        "pred": "synthetic" if resp["is_synthetic"] else "human",
        "confidence": resp["confidence"]
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(val)} procesados...")

r = pd.DataFrame(resultados)

# --- Matriz de confusión ---
tp = len(r[(r.real == "synthetic") & (r.pred == "synthetic")])
tn = len(r[(r.real == "human")    & (r.pred == "human")])
fp = len(r[(r.real == "human")    & (r.pred == "synthetic")])  # falso positivo
fn = len(r[(r.real == "synthetic") & (r.pred == "human")])     # falso negativo

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(f"                 Predicho: human  Predicho: synthetic")
print(f"Real: human           {tn:>4}              {fp:>4}   <- Falsos positivos")
print(f"Real: synthetic       {fn:>4}              {tp:>4}")
print(f"\nFalsos positivos (humano marcado como sintético): {fp}")
print(f"Falsos negativos (sintético marcado como humano): {fn}")
print(f"Precisión total: {(tp+tn)/len(r)*100:.1f}%")

# --- FAR y FRR ---
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\n=== FAR / FRR ===")
print(f"FAR (falsa aceptación - sintético pasa como humano): {far*100:.2f}%")
print(f"FRR (falso rechazo   - humano bloqueado):            {frr*100:.2f}%")

# --- Distribución de confidence ---
human_conf     = r[r.real == "human"]["confidence"].values
synthetic_conf = r[r.real == "synthetic"]["confidence"].values
print(f"\n=== DISTRIBUCIÓN DE CONFIDENCE ===")
print(f"Humanos    → media={human_conf.mean():.3f}  std={human_conf.std():.3f}  "
      f"min={human_conf.min():.3f}  max={human_conf.max():.3f}")
print(f"Sintéticos → media={synthetic_conf.mean():.3f}  std={synthetic_conf.std():.3f}  "
      f"min={synthetic_conf.min():.3f}  max={synthetic_conf.max():.3f}")

# --- Los casos más dudosos (cerca de 0.5) ---
r["distancia_umbral"] = (r["confidence"] - 0.5).abs()
dudosos = r.nsmallest(5, "distancia_umbral")
print(f"\n=== 5 CASOS MÁS DUDOSOS (confidence más cercana a 0.5) ===")
print(f"{'id':<20} {'real':<12} {'pred':<12} {'confidence'}")
print("-" * 60)
for _, row in dudosos.iterrows():
    emoji = "✅" if row.real == row.pred else "❌"
    print(f"{row.id[:18]:<20} {row.real:<12} {row.pred:<12} {row.confidence:.4f}  {emoji}")

# --- Errores concretos ---
errores = r[r.real != r.pred]
if len(errores) == 0:
    print(f"\n✅ Cero errores en {len(r)} audios de val.")
    print("   (Recuerda: el test oculto tiene hablantes y engines NUNCA vistos)")
else:
    print(f"\n❌ {len(errores)} errores:")
    for _, row in errores.iterrows():
        print(f"   {row.id} → real={row.real} pred={row.pred} conf={row.confidence:.4f}")