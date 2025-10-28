#!/usr/bin/env python3
# Requisitos: pip install opencv-python numpy
# NO guardes este archivo como "json.py"

import cv2
import numpy as np
import json
from pathlib import Path
import math

# --- Config hardcodeada ---
INPUT_DIR = Path("out")        # carpeta con imágenes
OUTPUT_DIR = Path("outjson")   # carpeta para JSONs
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Recorrer imágenes ---
images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
print(f"Encontradas {len(images)} imágenes")

for path in images:
    print(f"Procesando: {path.name}")
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        print(f"  ❌ No se pudo leer {path}")
        continue

    # Asegurar binaria {0,255} (por si vienen en escala de grises)
    _, bin_img = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)

    # Buscar contornos y quedarnos con el mayor (una sola pieza)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        data = {"image": path.name, "ok": False, "msg": "sin contornos"}
        with open(OUTPUT_DIR / f"{path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        continue

    cnt = max(contours, key=cv2.contourArea)

    # Métricas básicas
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

    # Bounding boxes
    x, y, w, h = cv2.boundingRect(cnt)
    extent = float(area / (w * h)) if w > 0 and h > 0 else 0.0

    # Rectángulo mínimo ≈ Feret mayor/menor + orientación
    (cx, cy), (Wm, Hm), angle = cv2.minAreaRect(cnt)  # angle en [-90,0)
    feret_max = float(max(Wm, Hm)) if Wm > 0 and Hm > 0 else 0.0
    feret_min = float(min(Wm, Hm)) if Wm > 0 and Hm > 0 else 1e-6
    aspect_ratio = float(feret_max / feret_min) if feret_min > 0 else 0.0
    orientation_deg = float(angle)

    # Casco convexo → solidity
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Momentos de Hu
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M).flatten().astype(float).tolist()

    # --- NUEVOS PARÁMETROS ---
    # 1) Cantidad de segmentos lineales (nº de lados) por aproximación poligonal
    eps = 0.01 * perimeter if perimeter > 0 else 1.0
    approx = cv2.approxPolyDP(cnt, eps, True)
    line_segments = int(len(approx))

    # 2) (Ya incluido) aspect_ratio = feret_max / feret_min

    # 3) Área / área del círculo cuyo diámetro = mayor dimensión (feret_max)
    circle_area = math.pi * (feret_max * 0.5) ** 2 if feret_max > 0 else 0.0
    area_to_circle_diam_fmax = float(area / circle_area) if circle_area > 0 else 0.0

    # Empaquetar y guardar
    data = {
        "image": path.name,
        "ok": True,
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "extent": extent,
        "feret_max": feret_max,
        "feret_min": feret_min,
        "aspect_ratio": aspect_ratio,
        "orientation_deg": orientation_deg,
        "solidity": solidity,
        "hu_moments": hu,
        # nuevos
        "line_segments": line_segments,
        "area_to_circle_diam_fmax": area_to_circle_diam_fmax
    }

    with open(OUTPUT_DIR / f"{path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Listo. JSONs escritos en:", OUTPUT_DIR)
