#!/usr/bin/env python3
import sys
import cv2
import numpy as np

# Fracción del perímetro para epsilon de RDP
EPS_FRAC = 0.05  # subí para menos vértices, bajá para más detalle

def _polygon_interior_angles_deg(poly: np.ndarray) -> np.ndarray:
    """
    poly: Nx1x2 o Nx2. Devuelve array con los ángulos interiores (grados).
    """
    P = poly.reshape(-1, 2).astype(np.float64)
    n = len(P)
    if n < 3:
        return np.array([], dtype=np.float64)

    def angle(a, b, c):
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return np.nan
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    angs = []
    for i in range(n):
        a = P[(i - 1) % n]
        b = P[i]
        c = P[(i + 1) % n]
        angs.append(angle(a, b, c))
    return np.array(angs, dtype=np.float64)

def main(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ No se pudo abrir: {path}")
        sys.exit(1)

    # Asegurar binaria pura {0,255} sin cambiar polaridad
    _, bin0 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Objeto en blanco: si el fondo domina en blancos, invertimos
    need_invert = cv2.countNonZero(bin0) > (bin0.size // 2)
    bin_obj = cv2.bitwise_not(bin0) if need_invert else bin0

    # QUEDARSE con la mayor componente (el objeto) para evitar ruido
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_obj, 8)
    if num <= 1:
        print("❌ No se encontraron componentes.")
        sys.exit(1)
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == idx).astype(np.uint8) * 255

    # Contornos con jerarquía para contar agujeros internos
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or hier is None:
        print("❌ No se encontraron contornos.")
        sys.exit(1)

    # Contorno externo principal (padre de mayor área)
    parent_idx = max(
        [i for i,(p,_,_,_) in enumerate(hier[0]) if p == -1],
        key=lambda i: cv2.contourArea(cnts[i]),
        default=None
    )
    if parent_idx is None:
        print("❌ No se encontró contorno externo principal.")
        sys.exit(1)

    cnt = cnts[parent_idx]
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))

    # Hu
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    hu_log = [-np.sign(v) * np.log10(abs(v) + 1e-30) for v in hu]

    # BBoxes y ratios
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = (w / h) if h > 0 else 0.0
    rect = cv2.minAreaRect(cnt)               # (center,(W,H),angle)
    rw, rh = rect[1]
    rotated_aspect = (max(rw, rh) / max(1e-6, min(rw, rh))) if min(rw, rh) > 0 else 0.0

    # Circularidad
    circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0

    # Extent (ocupación del bbox axis-aligned)
    extent = (area / (w * h)) if (w > 0 and h > 0) else 0.0

    # Solidez (área / área del casco convexo)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    hull_vertices = len(hull)

    # Eccentricidad (elipse ajustada) — requiere >=5 puntos
    if len(cnt) >= 5:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(cnt)  # mayor, menor
        a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (b*b)/(a*a)))) if a > 0 else 0.0
        ellipse_angle = float(angle)
    else:
        eccentricity = float('nan')
        ellipse_angle = float('nan')

    # Rugosidad radial del contorno (std de distancia al centroide)
    cx = m['m10']/m['m00'] if m['m00'] != 0 else (x + w/2.0)
    cy = m['m01']/m['m00'] if m['m00'] != 0 else (y + h/2.0)
    dists = np.sqrt((cnt[:,0,0] - cx)**2 + (cnt[:,0,1] - cy)**2)
    roughness_std = float(dists.std()) if dists.size else 0.0

    # Agujeros internos = hijos del contorno padre en la jerarquía
    holes_idxs = [i for i,(_,parent,_,_) in enumerate(hier[0]) if parent == parent_idx]
    holes_area = sum(cv2.contourArea(cnts[i]) for i in holes_idxs)
    holes_ratio = (holes_area / area) if area > 0 else 0.0

    # ======= Aproximación Ramer–Douglas–Peucker (RDP) =======
    rdp_eps = float(EPS_FRAC * perim)
    approx = cv2.approxPolyDP(cnt, rdp_eps, True)
    tries = 0
    while len(approx) < 3 and tries < 3:
        rdp_eps *= 0.5
        approx = cv2.approxPolyDP(cnt, rdp_eps, True)
        tries += 1

    rdp_vertices = len(approx)
    rdp_perim = cv2.arcLength(approx, True) if rdp_vertices >= 2 else np.nan
    rdp_area  = cv2.contourArea(approx) if rdp_vertices >= 3 else np.nan
    rdp_perimeter_ratio = float(rdp_perim / perim) if perim > 0 and not np.isnan(rdp_perim) else np.nan
    rdp_area_ratio      = float(rdp_area / area)  if area  > 0 and not np.isnan(rdp_area)  else np.nan

    angles = _polygon_interior_angles_deg(approx)
    angles_mean = float(np.nanmean(angles)) if angles.size else np.nan
    angles_std  = float(np.nanstd(angles))  if angles.size else np.nan

    # ---- Prints ----
    print(f"\n===== Archivo: {path} =====")
    print("=== Parámetros del objeto ===")
    print(f"Área (px^2):                 {area:.2f}")
    print(f"Perímetro (px):              {perim:.2f}")
    print(f"Aspect ratio (bbox w/h):     {aspect_ratio:.3f}")
    print(f"Rotated aspect (minRect):    {rotated_aspect:.3f}")
    print(f"Circularidad:                {circularity:.3f}   (1=círculo)")
    print(f"Extent (área/bbox):          {extent:.3f}")
    print(f"Solidity (área/hull):        {solidity:.3f}")
    print(f"Hull vértices:               {hull_vertices}")
    print(f"Eccentricidad (elipse):      {eccentricity:.3f}")
    print(f"Ángulo elipse (deg):         {ellipse_angle:.1f}")
    print(f"Rugosidad radial (std px):   {roughness_std:.3f}")

    print("\nHu invariants:")
    for i, v in enumerate(hu, start=1):
        print(f"  Hu[{i}]: {v:.6e}")
    print("Hu invariants (log10, signo):")
    for i, v in enumerate(hu_log, start=1):
        print(f"  Hu_log[{i}]: {v:+.3f}")

    print("\n=== Agujeros internos (vía jerarquía) ===")
    print(f"Número de agujeros:          {len(holes_idxs)}")
    print(f"Área total agujeros:         {holes_area:.2f} px")
    print(f"Relación área agujero/obj.:  {holes_ratio:.3f}")

    print("\n=== Aproximación RDP (cv2.approxPolyDP) ===")
    print(f"epsilon (px):                {rdp_eps:.2f}  (EPS_FRAC={EPS_FRAC:.3f})")
    print(f"vértices RDP:                {rdp_vertices}")
    print(f"perímetro_polígono/orig:     {rdp_perimeter_ratio:.3f}")
    print(f"área_polígono/orig:          {rdp_area_ratio:.3f}")
    print(f"ángulos interiores (deg):    media={angles_mean:.1f}, std={angles_std:.1f}")

if __name__ == "__main__":
    # Cambiá los paths si querés procesar varios archivos
    paths = [
        "resultados_preprocesados/IMG_20251025_132845/04_morph_open.png",
        "resultados_preprocesados/IMG_20251025_132912/04_binary_morph_open_norm.png",
        "resultados_preprocesados/IMG_20251025_133013/04_morph_open.png",
        "resultados_preprocesados/IMG_20251025_133527/04_morph_open.png",
    ]
    for p in paths:
        main(p)
