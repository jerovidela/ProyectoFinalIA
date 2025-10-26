#!/usr/bin/env python3
import cv2
import json
import numpy as np
from pathlib import Path

# ==== CONFIGURACIÓN (editá estas rutas) ====
INPUT_DIR  = Path("outes")
OUTPUT_DIR = Path("resultados_json")
EPS_FRAC   = 0.05   # fracción del perímetro usada en approxPolyDP
# ============================================

def polygon_interior_angles_deg(poly):
    P = poly.reshape(-1, 2).astype(float)
    n = len(P)
    if n < 3: return []
    angs = []
    for i in range(n):
        a, b, c = P[i-1], P[i], P[(i+1) % n]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angs.append(np.degrees(np.arccos(cosang)))
    return angs

def analyze_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ No se pudo abrir: {path.name}")
        return None

    _, bin0 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(bin0) > bin0.size // 2:
        bin0 = cv2.bitwise_not(bin0)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin0, 8)
    if num <= 1:
        print(f"❌ Sin componentes en {path.name}")
        return None

    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == idx).astype(np.uint8) * 255

    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or hier is None:
        print(f"❌ Sin contornos en {path.name}")
        return None

    parent_idx = max([i for i,(p,_,_,_) in enumerate(hier[0]) if p == -1],
                     key=lambda i: cv2.contourArea(cnts[i]))
    cnt = cnts[parent_idx]

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)

    # Hu invariants
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten().tolist()

    # Aproximación poligonal
    eps = EPS_FRAC * perim
    approx = cv2.approxPolyDP(cnt, eps, True)
    angs = polygon_interior_angles_deg(approx)

    return {
        "file": path.name,
        "area_px2": area,
        "perimeter_px": perim,
        "bbox_w": w, "bbox_h": h,
        "vertices": len(approx),
        "angles_mean_deg": float(np.mean(angs)) if angs else None,
        "angles_std_deg": float(np.std(angs)) if angs else None,
        "hu": hu,
    }

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    images = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")]
    print(f"Encontradas {len(images)} imágenes")

    for img_path in images:
        data = analyze_image(img_path)
        if data:
            out_path = OUTPUT_DIR / f"{img_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ {img_path.name} → {out_path.name}")

if __name__ == "__main__":
    main()
