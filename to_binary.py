import cv2
import numpy as np
from pathlib import Path

input_dir  = Path("img")
output_dir = Path("out")
output_dir.mkdir(exist_ok=True)

for path in sorted(input_dir.glob("*.jpg")):
    print(f"Procesando: {path.name}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h0, w0 = img.shape[:2]
    new_w = 640
    new_h = int(h0 * new_w / w0)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(img, (5,5), 0)

    binary_gray = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 5
    )
    binary = binary_gray if cv2.countNonZero(binary_gray) < (binary_gray.size // 2) else cv2.bitwise_not(binary_gray)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == idx, 255, 0).astype(np.uint8)

    h, w = binary.shape
    inv = 255 - binary
    ff = inv.copy()
    mask_ff = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0,0), 0)
    holes  = ff
    filled = cv2.bitwise_or(binary, holes)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    gx = cv2.Scharr(blur, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("No se hallaron contornos. Guardo edges como referencia.")
        cv2.imwrite(str(output_dir / path.name), edges)
        continue

    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    fused = cv2.bitwise_or(solid, filled)

    out_path = output_dir / path.name
    cv2.imwrite(str(out_path), fused)
    print(f"Guardado en: {out_path}")

print("\n Procesamiento completo.")
