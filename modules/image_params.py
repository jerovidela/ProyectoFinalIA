import cv2
import json
import numpy as np
from pathlib import Path

def extract_image_features(input_dir="out", output_dir="outjson", eps_frac=0.02, k_curv=5):
    """
    Extrae características geométricas de imágenes binarias y las guarda en archivos JSON.

    Args:
        input_dir (str | Path): Carpeta con imágenes binarias (.png, .jpg, etc.)
        output_dir (str | Path): Carpeta donde se guardarán los .json de salida.
        eps_frac (float): Fracción del perímetro usada para simplificar contornos.
        k_curv (int): Cantidad de puntos de separación para estimar curvatura local.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Recolectar imágenes
    images = []
    for e in IMG_EXTS:
        images += list(input_dir.glob(f"*{e}"))
        images += list(input_dir.glob(f"*{e.upper()}"))

    if not images:
        print(f"⚠️ No se encontraron imágenes en {input_dir}")
        return

    for img_path in sorted(images):
        g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"⚠️ No se pudo leer la imagen {img_path.name}")
            continue

        # Binarizar
        _, bin_img = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(bin_img) > bin_img.size // 2:
            bin_img = cv2.bitwise_not(bin_img)

        # Extraer componente principal
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
        if num_labels <= 1:
            print(f"⚠️ No se encontró objeto en {img_path.name}")
            continue
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8) * 255

        # Contorno principal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ No se encontró contorno en {img_path.name}")
            continue
        contour = max(contours, key=cv2.contourArea)

        # --- Cálculo de características ---
        area = cv2.contourArea(contour)
        (circ_cx, circ_cy), circle_radius = cv2.minEnclosingCircle(contour)
        circle_area_ratio = float(area / (np.pi * (circle_radius ** 2))) if circle_radius > 0 else 0.0

        # Momentos de Hu
        M = cv2.moments(contour)
        hu_raw = cv2.HuMoments(M).flatten()

        def transform_hu(v):
            return 0.0 if v == 0 else float(-np.sign(v) * np.log10(abs(v)))

        hu_moment_1 = transform_hu(hu_raw[0]) if hu_raw.size >= 1 else 0.0
        hu_moment_2 = transform_hu(hu_raw[1]) if hu_raw.size >= 2 else 0.0

        # Ángulo mínimo entre segmentos
        eps = eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = approx.reshape(-1, 2).astype(float)
        angles = []
        n = len(pts)
        if n >= 3:
            for i in range(n):
                p_prev = pts[i - 1]
                p = pts[i]
                p_next = pts[(i + 1) % n]
                v1, v2 = p_prev - p, p_next - p
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_a))
                else:
                    angle = 0.0
                angles.append(angle)
        angles_min = float(np.min(angles)) if angles else 0.0

        # Curvatura máxima
        pts_full = contour.reshape(-1, 2)
        n_full = len(pts_full)
        curv_vals = []
        if n_full >= 2 * k_curv + 1:
            for i in range(n_full):
                p1 = pts_full[(i - k_curv) % n_full]
                p2 = pts_full[i]
                p3 = pts_full[(i + k_curv) % n_full]
                v1, v2 = p2 - p1, p3 - p2
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    c = cross / (n1 * n2)
                    curv_vals.append(abs(c))
        curvature_max = float(np.max(curv_vals)) if curv_vals else 0.0

        # --- Guardar salida ---
        out = {
            "filename": img_path.name,
            "circle_area_ratio": float(circle_area_ratio),
            "hu_moment_1": float(hu_moment_1),
            "angles_min": float(angles_min),
            "hu_moment_2": float(hu_moment_2),
            "curvature_max": float(curvature_max),
        }

        out_path = output_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✅ Guardado → {out_path}")

    print("\n🏁 Extracción de características completa.")


# Ejemplo de uso:
# extract_image_features("out", "outjson")
