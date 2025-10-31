import cv2
import numpy as np
import os
import json
import glob

img_dir = "img"
output_dir = "outjson"

os.makedirs(output_dir, exist_ok=True)

image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    image_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))

for img_path in sorted(image_files):
    filename = os.path.basename(img_path)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ERROR: No se pudo cargar {filename}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"  ERROR: No se encontraron contornos en {filename}")
        continue
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        print(f"  ERROR: Contorno muy pequeño en {filename}")
        continue
    # ============================================================================
    # PARÁMETRO 1: CIRCLE_AREA_RATIO
    # ============================================================================
    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circle_area_ratio = area / circle_area if circle_area > 0 else 0
    # ============================================================================
    # PARÁMETRO 2: HU_MOMENT_1 (Segundo momento invariante de Hu)
    # PARÁMETRO 3: HU_MOMENT_2 (Tercer momento invariante de Hu)
    # ============================================================================
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moment_1 = abs(hu_moments[1]) if len(hu_moments) > 1 else 0
    hu_moment_2 = abs(hu_moments[2]) if len(hu_moments) > 2 else 0
    # ============================================================================
    # PARÁMETRO 4: ANGLES_MIN
    # ============================================================================
    angles = []
    n_points = len(contour)
    for i in range(n_points):
        p1 = contour[(i-1) % n_points][0]
        p2 = contour[i][0]
        p3 = contour[(i+1) % n_points][0]
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))
    angles_min = min(angles) if angles else 180
    # ============================================================================
    # PARÁMETRO 5: CURVATURE_MAX
    # ============================================================================
    curvatures = []
    for i in range(n_points):
        if i < 2 or i >= n_points - 2:
            continue            
        p1 = contour[i-2][0]
        p2 = contour[i-1][0]
        p3 = contour[i][0]
        p4 = contour[i+1][0]
        p5 = contour[i+2][0]
        dx1 = (p4[0] - p2[0]) / 2.0
        dy1 = (p4[1] - p2[1]) / 2.0
        dx2 = p4[0] - 2*p3[0] + p2[0]
        dy2 = p4[1] - 2*p3[1] + p2[1]
        numerator = abs(dx1 * dy2 - dy1 * dx2)
        denominator = (dx1**2 + dy1**2)**(3/2)
        if denominator > 1e-6:
            curvature = numerator / denominator
            curvatures.append(curvature)
    curvature_max = max(curvatures) if curvatures else 0   
    result = {
        "filename": filename,
        "circle_area_ratio": float(circle_area_ratio),
        "hu_moment_1": float(hu_moment_1),
        "angles_min": float(angles_min),
        "hu_moment_2": float(hu_moment_2),
        "curvature_max": float(curvature_max)
    }
    base_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Guardado en: {output_file}")