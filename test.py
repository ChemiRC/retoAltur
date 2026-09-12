import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64, json
import pandas as pd
import requests

URL = "http://127.0.0.1:8000/detect"
df = pd.read_csv("manifest.csv")

# Toma 5 humanos y 5 sintéticos del val
val = df[df["split"] == "val"]
muestra = pd.concat([
    val[val["label"] == "human"].head(5),
    val[val["label"] == "synthetic"].head(5)
])

ok, total = 0, 0
print(f"{'archivo':<20} {'real':<12} {'predicho':<12} {'confidence':<12} {'resultado'}")
print("-" * 70)

for _, row in muestra.iterrows():
    path = os.path.join("audio", row["anon_id"] + ".wav")
    if not os.path.exists(path):
        continue

    b64 = base64.b64encode(open(path, "rb").read()).decode()
    resp = requests.post(URL, json={"audio_base64": b64}).json()

    pred = "synthetic" if resp["is_synthetic"] else "human"
    conf = resp["confidence"]
    acierto = pred == row["label"]
    ok += acierto
    total += 1

    emoji = "✅" if acierto else "❌"
    print(f"{row['anon_id'][:18]:<20} {row['label']:<12} {pred:<12} {conf:<12} {emoji}")

print("-" * 70)
print(f"Resultado: {ok}/{total} correctos")