#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # para guardar/cargar el modelo

# --- CONFIGURACIÓN ---
train_dir = Path("outjson")   # carpeta con JSONs de entrenamiento
model_path = Path("kmeans_model.pkl")  # archivo donde se guarda el modelo

# --- CARGA DE DATOS ---
data = []
names = []

for path in sorted(train_dir.glob("*.json")):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # elegimos las features numéricas
    feats = [
        j["area"],
        j["perimeter"],
        j["circularity"],
        j["aspect_ratio"],
        j["solidity"],
        j["area_to_circle_diam_fmax"],
        j["line_segments"]
    ]
    data.append(feats)
    names.append(j["image"])

X = np.array(data)
print(f"Se cargaron {len(X)} muestras")

# --- NORMALIZAR Y ENTRENAR ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=0)  # podés ajustar n_clusters
kmeans.fit(X_scaled)

# --- GUARDAR MODELO ---
joblib.dump((scaler, kmeans), model_path)
print("Modelo guardado en", model_path)

# --- PREDICCIÓN DE UN NUEVO OBJETO ---
json_path = "Tuerca10.json"
with open(json_path, "r", encoding="utf-8") as f:
    new = json.load(f)

new_feats = np.array([[
    new["area"],
    new["perimeter"],
    new["circularity"],
    new["aspect_ratio"],
    new["solidity"],
    new["area_to_circle_diam_fmax"],
    new["line_segments"]
]])

new_scaled = scaler.transform(new_feats)
cluster = kmeans.predict(new_scaled)[0]
print(f"La imagen '{new['image']}' pertenece al grupo #{cluster}")

labels = kmeans.labels_

for c in np.unique(labels):
    print(f"\nCluster {c}:")
    for name in np.array(names)[labels == c]:
        print(" ", name)

