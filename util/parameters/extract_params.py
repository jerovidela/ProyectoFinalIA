import cv2
import json
import numpy as np
from pathlib import Path

INPUT_DIR = Path("out")  
OUTPUT_DIR = Path("outjson")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

EPS_FRAC = 0.02
K_CURV = 5

images = []
for e in IMG_EXTS:
    images += list(INPUT_DIR.glob(f"*{e}"))
    images += list(INPUT_DIR.glob(f"*{e.upper()}"))

if not images:
    print(f"No se encuentran imagenes en {INPUT_DIR}")

for img_path in sorted(images):

    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        print("No se puede leer imagen")
        continue

    _, bin_img = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(bin_img) > bin_img.size // 2:
        bin_img = cv2.bitwise_not(bin_img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
    if num_labels <= 1:
        print("No se encontro objeto")
        continue
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No se encontró contorno")
        continue
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    (circ_cx, circ_cy), circle_radius = cv2.minEnclosingCircle(contour)
    if circle_radius > 0:
        circle_area_ratio = float(area / (np.pi * (circle_radius ** 2)))
    else:
        circle_area_ratio = 0.0

    M = cv2.moments(contour)
    hu_raw = cv2.HuMoments(M).flatten()
    def transform_hu(v):
        if v == 0:
            return 0.0
        return float(-np.sign(v) * np.log10(abs(v)))
    hu_moment_1 = transform_hu(hu_raw[0]) if hu_raw.size >= 1 else 0.0
    hu_moment_2 = transform_hu(hu_raw[1]) if hu_raw.size >= 2 else 0.0

    eps = EPS_FRAC * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2).astype(float)
    angles = []
    n = len(pts)
    if n >= 3:
        for i in range(n):
            p_prev = pts[i - 1]
            p = pts[i]
            p_next = pts[(i + 1) % n]

            v1 = p_prev - p
            v2 = p_next - p
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_a))
            else:
                angle = 0.0
            angles.append(angle)
    angles_min = float(np.min(angles)) if angles else 0.0

    pts_full = contour.reshape(-1, 2)
    n_full = len(pts_full)
    curv_vals = []
    if n_full >= 2 * K_CURV + 1:
        for i in range(n_full):
            p1 = pts_full[(i - K_CURV) % n_full]
            p2 = pts_full[i]
            p3 = pts_full[(i + K_CURV) % n_full]
            v1 = p2 - p1
            v2 = p3 - p2
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                c = cross / (n1 * n2)
                curv_vals.append(abs(c))
    curvature_max = float(np.max(curv_vals)) if curv_vals else 0.0

    out = {
        "filename": img_path.name,
        "circle_area_ratio": float(circle_area_ratio),
        "hu_moment_1": float(hu_moment_1),
        "angles_min": float(angles_min),
        "hu_moment_2": float(hu_moment_2),
        "curvature_max": float(curvature_max),
    }

    out_path = OUTPUT_DIR / f"{img_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")