#!/usr/bin/env python3
# Requisitos: Python 3.8+ (solo stdlib)
import json
from pathlib import Path

IN_DIR = Path("outjson")   # carpeta con los JSONs que generaste antes

files = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() == ".json"])
print(f"Encontrados {len(files)} JSONs")

for jf in files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"{jf.name}: error leyendo JSON -> {e}")
        continue

    # Levantar features con defaults por si faltan
    circ = float(d.get("circularity", 0.0))
    ar   = float(d.get("aspect_ratio", 1.0))  # feret_max/feret_min
    sol  = float(d.get("solidity", 1.0))
    segs = d.get("line_segments", None)
    ratC = d.get("area_to_circle_diam_fmax", None)

    # --- Reglas orientativas (muy simples) ---
    # 1) Alargado vs compacto
    ELONG = ar >= 2.2

    pred = "desconocido"

    if ELONG:
        # Tornillo vs clavo por “irregularidad” (solidity) y circularidad muy baja
        # Tornillo tiende a tener rosca/cabeza -> menor solidity
        if sol < 0.84 or circ < 0.20:
            pred = "tornillo"
        else:
            pred = "clavo"
    else:
        # Compactos: arandela (disco sólido) vs tuerca (hexágono sólido)
        # Usamos la razón área / área del círculo de D=feret_max:
        #  - Disco ideal  ~1.00
        #  - Hexágono reg ~0.83
        if ratC is not None:
            if ratC >= 0.92 and circ >= 0.80:
                pred = "arandela"
            elif 0.70 <= ratC <= 0.90 and segs is not None and 5 <= int(segs) <= 7:
                pred = "tuerca"
            else:
                # Fallback por circularidad y nº de lados
                if circ >= 0.83:
                    pred = "arandela"
                elif segs is not None and 5 <= int(segs) <= 7:
                    pred = "tuerca"
                else:
                    pred = "arandela" if circ >= 0.75 else "tuerca"
        else:
            # Sin ratC: usar circularidad y lados
            if circ >= 0.83:
                pred = "arandela"
            elif segs is not None and 5 <= int(segs) <= 7:
                pred = "tuerca"
            else:
                pred = "arandela" if circ >= 0.75 else "tuerca"

    print(f"{jf.name}: {pred}")
