#!/usr/bin/env python3
# Requisitos: pip install opencv-python numpy
# ¡No guardes este archivo como "json.py"!

import cv2
import numpy as np
import json
from pathlib import Path

# --- CONFIG HARD-CODEADA ---
INPUT_DIR = Path("out")        # carpeta de entrada (binarias 0/255)
OUTPUT_DIR = Path("outjson")   # carpeta de salida
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- RECORRER IMÁGENES ---
images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
print(f"Encontradas {len(images)} imágenes")

for path in images:
    print(f"Procesando: {path.name}")

    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        print(f"  ❌ No se pudo leer {path}")
        continue

    # Asegurar binaria {0,255} y objeto=blanco
    _, bin0 = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(bin0) < (bin0.size // 2):
        # fondo blanco, invertimos para dejar objeto=blanco
        bin_img = cv2.bitwise_not(bin0)
    else:
        bin_img = bin0

    H, W = bin_img.shape[:2]

    # Contornos con jerarquía (no hacemos ningún umbral ni inversión)
    contours, hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(contours) == 0:
        print("  ⚠️ Sin contornos.")
        data = {"image": path.name, "objects": []}
        with open(OUTPUT_DIR / f"{path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        continue

    hier = hier[0]
    objects = []
    obj_id = 0

    for i, cnt in enumerate(contours):
        # Solo contornos externos (padre = -1)
        if hier[i][3] != -1:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Descarta el "marco" si toca borde (margen 0 px)
        if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
            continue

        # Geometría básica
        area_ext = float(cv2.contourArea(cnt))
        perim_ext = float(cv2.arcLength(cnt, True))
        area_holes = 0.0  # en tu dataset no usamos huecos para clasificar
        perim_holes = 0.0
        area_solid = area_ext
        perim_total = perim_ext

        # BBoxes y rectángulo mínimo (Feret aprox)
        bbox_area = float(w * h)
        rect = cv2.minAreaRect(cnt)       # ((cx,cy),(W,H),angle)
        (cx, cy), (Wm, Hm), angle = rect
        rect_area = float(Wm * Hm if Wm > 0 and Hm > 0 else 0.0)
        feret_max = float(max(Wm, Hm)) if Wm > 0 and Hm > 0 else 0.0
        feret_min = float(min(Wm, Hm)) if Wm > 0 and Hm > 0 else 0.0
        feret_ratio = float(feret_max / (feret_min + 1e-6))

        # Casco convexo y métricas asociadas
        hull_pts = cv2.convexHull(cnt, returnPoints=True)
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        hull_area = float(cv2.contourArea(hull_pts)) if hull_pts is not None else 0.0
        hull_perim = float(cv2.arcLength(hull_pts, True)) if hull_pts is not None else 0.0

        solidity = float(area_solid / (hull_area + 1e-6)) if hull_area > 0 else 0.0
        convexity = float((hull_perim + 1e-6) / (perim_ext + 1e-6)) if perim_ext > 0 else 0.0

        # Circularidad / rugosidad
        circularity = float(4.0 * np.pi * area_solid / ((perim_total + 1e-6) ** 2)) if area_solid > 0 else 0.0
        roughness = float(((perim_total + 1e-6) ** 2) / (4.0 * np.pi * (area_solid + 1e-6))) if area_solid > 0 else 0.0

        # Extent / rectangularidad / orientación
        extent = float(area_solid / (bbox_area + 1e-6)) if bbox_area > 0 else 0.0
        rectangularity = float(area_solid / (rect_area + 1e-6)) if rect_area > 0 else 0.0
        orientation_deg = float(angle)  # [-90,0)

        # Defectos de convexidad (hexágonos y rosca generan defectos visibles)
        defects_count = 0
        defects_mean_depth = 0.0
        if hull_idx is not None and len(hull_idx) >= 3 and len(cnt) >= 4:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and len(defects) > 0:
                defects_count = int(defects.shape[0])
                depths = defects[:, 0, 3].astype(np.float32) / 256.0
                defects_mean_depth = float(np.mean(depths))

        # Aproximación poligonal (útil para “hexágono sólido”)
        eps = 0.01 * perim_ext if perim_ext > 0 else 1.0
        approx = cv2.approxPolyDP(cnt, eps, True)
        approx_vertices = int(len(approx))

        # Momentos de Hu
        m = cv2.moments(cnt)
        hu = cv2.HuMoments(m).flatten().astype(float).tolist()

        obj = {
            "object_id": obj_id,
            "area_external": area_ext,
            "area_holes": area_holes,
            "area_solid": area_solid,
            "perimeter_external": perim_ext,
            "perimeter_holes": perim_holes,
            "perimeter_total": perim_total,
            "circularity": circularity,
            "roughness": roughness,
            "bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(w), "bbox_h": int(h),
            "extent": extent,
            "rect_area": rect_area,
            "rectangularity": rectangularity,
            "orientation_deg": orientation_deg,
            "feret_min": feret_min,
            "feret_max": feret_max,
            "feret_ratio": feret_ratio,
            "aspect_ratio": feret_ratio,
            "hull_area": hull_area,
            "hull_perimeter": hull_perim,
            "solidity": solidity,
            "convexity": convexity,
            "convexity_defects_count": defects_count,
            "convexity_defects_mean_depth": defects_mean_depth,
            "approx_vertices": approx_vertices,
            "hu_moments": hu
        }
        objects.append(obj)
        obj_id += 1

    data = {"image": path.name, "objects": objects}
    with open(OUTPUT_DIR / f"{path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Listo. JSONs escritos en:", OUTPUT_DIR)
