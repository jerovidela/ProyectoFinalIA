#!/usr/bin/env python3
# Requisitos: pip install opencv-python numpy
# NOTA: no guardes este archivo como "json.py"

import cv2
import numpy as np
import json
from pathlib import Path

# --- CONFIGURACIÓN HARD-CODEADA ---
INPUT_DIR = Path("out")        # carpeta de entrada
OUTPUT_DIR = Path("outjson")   # carpeta de salida de JSONs
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

    # Contornos con jerarquía para detectar huecos (hijos)
    contours, hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(contours) == 0:
        print("  ⚠️ Sin contornos detectados.")
        data = {"image": path.name, "objects": []}
        out_file = OUTPUT_DIR / f"{path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        continue

    hier = hier[0]  # (N,4): [next, prev, first_child, parent]
    objects = []
    obj_id = 0

    for i, cnt in enumerate(contours):
        # Solo contornos externos (sin padre)
        if hier[i][3] != -1:
            continue

        # Hijos (huecos)
        holes_idx = []
        child = hier[i][2]
        while child != -1:
            holes_idx.append(child)
            child = hier[child][0]  # siguiente hermano

        # Geometría básica
        area_ext = float(cv2.contourArea(cnt))  # área del contorno externo
        perim_ext = float(cv2.arcLength(cnt, True))

        area_holes = 0.0
        perim_holes = 0.0
        for h in holes_idx:
            ah = float(cv2.contourArea(contours[h]))
            ph = float(cv2.arcLength(contours[h], True))
            area_holes += abs(ah)
            perim_holes += ph

        area_solid = max(area_ext - area_holes, 0.0)
        perim_total = perim_ext + perim_holes

        # Bounding boxes
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = float(w * h)
        rect = cv2.minAreaRect(cnt)   # ((cx,cy),(W,H),angle[-90,0))
        (cx, cy), (W, H), angle = rect
        rect_area = float(W * H)

        # Feret (aprox con minAreaRect)
        feret_max = float(max(W, H))
        feret_min = float(min(W, H)) if min(W, H) > 0 else 1e-6
        feret_ratio = float(feret_max / feret_min)

        # Hull (casco convexo)
        hull_pts = cv2.convexHull(cnt, returnPoints=True)
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        hull_area = float(cv2.contourArea(hull_pts)) if hull_pts is not None else 0.0
        hull_perim = float(cv2.arcLength(hull_pts, True)) if hull_pts is not None else 0.0

        solidity = float(area_solid / hull_area) if hull_area > 0 else 0.0
        convexity = float(hull_perim / perim_ext) if perim_ext > 0 else 0.0

        # Circularidad y rugosidad (usar perímetro total para “anillos”)
        circularity = float(4.0 * np.pi * area_solid / (perim_total**2)) if perim_total > 0 and area_solid > 0 else 0.0
        roughness = float((perim_total**2) / (4.0 * np.pi * area_solid)) if area_solid > 0 else 0.0

        # Extent y rectangularidad
        extent = float(area_solid / bbox_area) if bbox_area > 0 else 0.0
        rectangularity = float(area_solid / rect_area) if rect_area > 0 else 0.0

        # Aspect ratio (de minAreaRect)
        aspect_ratio = float(feret_ratio)

        # Orientación (tal cual entrega minAreaRect)
        orientation_deg = float(angle)  # en grados, [-90, 0)

        # Número de huecos y fracción de área hueca
        holes_count = int(len(holes_idx))
        hole_fraction = float(area_holes / (area_ext + 1e-6)) if area_ext > 0 else 0.0

        # Defectos de convexidad (para tuercas con vértices, etc.)
        defects_count = 0
        defects_mean_depth = 0.0
        if hull_idx is not None and len(hull_idx) >= 3 and len(cnt) >= 4:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and len(defects) > 0:
                defects_count = int(defects.shape[0])
                # la profundidad viene *256
                depths = defects[:, 0, 3].astype(np.float32) / 256.0
                defects_mean_depth = float(np.mean(depths))

        # Aproximación poligonal (útil para tuerca ~hexágono)
        eps = 0.01 * perim_ext if perim_ext > 0 else 1.0
        approx = cv2.approxPolyDP(cnt, eps, True)
        approx_vertices = int(len(approx))

        # Momentos de Hu
        m = cv2.moments(cnt)
        hu = cv2.HuMoments(m).flatten().astype(float).tolist()

        # Euler por objeto (1 - #huecos)
        euler_number = int(1 - holes_count)

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
            "aspect_ratio": aspect_ratio,
            "hull_area": hull_area,
            "hull_perimeter": hull_perim,
            "solidity": solidity,
            "convexity": convexity,
            "holes_count": holes_count,
            "hole_fraction": hole_fraction,
            "convexity_defects_count": defects_count,
            "convexity_defects_mean_depth": defects_mean_depth,
            "approx_vertices": approx_vertices,
            "hu_moments": hu,
            "euler_number": euler_number
        }
        objects.append(obj)
        obj_id += 1

    data = {"image": path.name, "objects": objects}
    out_file = OUTPUT_DIR / f"{path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Listo. JSONs escritos en:", OUTPUT_DIR)
