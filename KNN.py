#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- CONFIGURACIÓN ---
train_dir = Path("outjson")    # carpeta con tus JSONs
model_path = Path("knn_model.pkl")  # modelo a guardar

# --- CARGA DE DATOS ---
data = []
labels = []

for path in sorted(train_dir.glob("*.json")):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    # Determinar la etiqueta según el nombre del archivo
    name = j["image"].lower()
    if "clavo" in name:
        label = "clavo"
    elif "tornillo" in name:
        label = "tornillo"
    elif "tuerca" in name:
        label = "tuerca"
    elif "arandela" in name:
        label = "arandela"
    else:
        continue  # ignorar si no se puede determinar

    # Features seleccionadas (7 en total)
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
    labels.append(label)

X = np.array(data)
y = np.array(labels)
print(f"Se cargaron {len(X)} muestras con etiquetas.")

# --- NORMALIZAR Y DIVIDIR ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)

# --- ENTRENAR KNN ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- EVALUAR ---
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {acc:.2f}")

# --- GUARDAR MODELO ---
joblib.dump((scaler, knn), model_path)
print("Modelo guardado en", model_path)

# --- PREDICCIÓN DE UNA NUEVA IMAGEN ---
json_path = "arandela01.json"  # JSON con la estructura del ejemplo Tornillo41
with open(json_path, "r", encoding="utf-8") as f:
    new = json.load(f)

new_feats = np.array([[new["area"], new["perimeter"], new["circularity"],
                       new["aspect_ratio"], new["solidity"],
                       new["area_to_circle_diam_fmax"], new["line_segments"]]])

new_scaled = scaler.transform(new_feats)
pred = knn.predict(new_scaled)[0]
print(f"La imagen '{new['image']}' fue clasificada como: {pred}")
