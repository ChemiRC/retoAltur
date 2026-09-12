# Altur Challenge — Detección de voz sintética en llamadas telefónicas

Sistema que, dada una llamada estéreo 8 kHz (canal 0 = llamante, canal 1 = agente),
decide si el llamante es humano o una voz sintética, y expone `POST /detect`.

## Enfoque

Combinamos **dos familias de señales** en un único clasificador
(`HistGradientBoosting` calibrado con isotónica):

1. **Acústicas (canal 0).** MFCC + deltas + estadísticos espectrales
   (centroid, bandwidth, rolloff, *flatness*, ZCR, RMS). El TTS tiende a ser
   espectralmente más suave/band-limited que la voz humana grabada por teléfono.
2. **Timing conversacional (canales 0 y 1).** A partir de un VAD por energía por
   canal medimos la latencia de respuesta del llamante tras cada turno del agente
   y, sobre todo, su **consistencia** (coeficiente de variación), además de
   solapamientos y pausas. Una máquina responde con timing robóticamente parejo;
   un humano, no. Esta señal es **independiente del engine de TTS**, así que es
   nuestro seguro contra voces sintéticas no vistas.

## Por qué no confiamos en el 100% de la validación

Con las features acústicas completas obtenemos AUC=1.0 / EER=0% en `val`
(hablantes disjuntos). **Eso es sospechoso, no un triunfo.** La importancia por
permutación muestra que una sola feature (`zcr_std`) separa las clases: un atajo
espectral del dataset. Lo probamos degradando `val` con códec telefónico G.711 +
ruido: el modelo de una sola feature colapsa a **EER 38%**, mientras que el modelo
con todas las features **se mantiene en EER 0%**. Conclusión: usamos el conjunto
completo (señal redundante y robusta) y añadimos **aumentación de canal** en
entrenamiento (`--augment`: mu-law, ruido, ganancia) para no aferrarnos al artefacto.
Umbral de operación: 0.5 (más robusto al set oculto que el 0.993 que sugiere el EER).

## Uso

```bash
pip install -r requirements.txt
# 1) Descomprimir altur-challenge-audio.zip en el repo (crea audio/)
# 2) Entrenar (genera model.pkl + threshold.json):
python train.py --data_dir . --augment
# 3) Servir:
uvicorn main:app --host 0.0.0.0 --port 8000
# exponer públicamente para el juez:  ngrok http 8000
```

`GET /health` para verificar que el modelo cargó. `POST /detect` acepta el base64
en varios nombres de campo o crudo en el body; nunca devuelve 500 al benchmark.

## Archivos

- `features.py` — extracción de features (acústicas + timing). Mismo código en train y serve.
- `augment.py` — degradaciones de canal para robustez.
- `train.py` — entrena con el split provisto, evalúa en val, reporta EER y features top.
- `main.py` — servidor FastAPI con `/detect` endurecido.