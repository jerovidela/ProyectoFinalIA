#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, argparse, sys
import numpy as np

# -------- args ----------
p = argparse.ArgumentParser(description="3D scatter de features de audio por PALABRA (color por palabra)")
p.add_argument("--save", default="scatter_3d_by_word.png", help="PNG de salida (además de mostrar)")
args = p.parse_args()

json_dir = "features_json"
use_norm = True

# -------- cargar datos ----------
pattern = os.path.join(json_dir, "*.json")
files = sorted(glob.glob(pattern))
files = [f for f in files if os.path.basename(f) != "mu_sd.json"]
if not files:
    print(f"No hay JSONs en {json_dir}")
    sys.exit(1)

X = []
WORDS = []
NAMES = []

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"ERR leyendo {fp}: {e}")
        continue

    # palabra desde el filename (antes del primer "_")
    fname = obj.get("filename", os.path.basename(fp))
    base = os.path.splitext(os.path.basename(fname))[0]
    palabra = base.split("_")[0].strip().lower() if "_" in base else base.lower()

    # features
    if use_norm and all(k in obj for k in ("zcr_std_z","rolloff95_std_z","mfcc_std_4_z")):
        feats = [float(obj["zcr_std_z"]), float(obj["rolloff95_std_z"]), float(obj["mfcc_std_4_z"])]
    else:
        # crudas (o normalizo al vuelo si hay mu_sd.json)
        if use_norm:
            ms_path = os.path.join(json_dir, "mu_sd.json")
            if os.path.exists(ms_path) and all(k in obj for k in ("zcr_std_z","rolloff95_std_z","mfcc_std_4_z")):
                with open(ms_path, "r", encoding="utf-8") as f:
                    ms = json.load(f)
                raw = np.array([float(obj["zcr_std_z"]), float(obj["rolloff95_std_z"]), float(obj["mfcc_std_4_z"])], dtype=float)
                mu = np.array(ms["mu"], dtype=float); sd = np.array(ms["sd"], dtype=float); sd[sd==0]=1.0
                feats = ((raw - mu) / sd).tolist()
            else:
                feats = [float(obj.get("zcr_std_z", 0.0)), float(obj.get("rolloff95_std_z", 0.0)), float(obj.get("mfcc_std_4_z", 0.0))]
        else:
            feats = [float(obj.get("zcr_std_z", 0.0)), float(obj.get("rolloff95_std_z", 0.0)), float(obj.get("mfcc_std_4_z", 0.0))]

    X.append(feats)
    WORDS.append(palabra)
    NAMES.append(base)

X = np.asarray(X, dtype=float)
WORDS = np.asarray(WORDS, dtype=object)

# -------- plot 3D ----------
import matplotlib
# intentar un backend interactivo si estás en venv sin GUI
if "agg" in matplotlib.get_backend().lower():
    for b in ("TkAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(b, force=True)
            break
        except Exception:
            pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection="3d")

# colores consistentes por palabra
palabras = sorted(list(set(WORDS.tolist())))
cmap_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
marker = "o"

for i, w in enumerate(palabras):
    mask = (WORDS == w)
    ax.scatter(X[mask,0], X[mask,1], X[mask,2],
               label=w, marker=marker, s=40, alpha=0.9,
               color=cmap_colors[i % len(cmap_colors)])

ax.set_xlabel("zcr_std_z" + ("_z" if use_norm else ""))
ax.set_ylabel("rolloff95_std_z" + ("_z" if use_norm else ""))
ax.set_zlabel("mfcc_std_4_z" + ("_z" if use_norm else ""))
ax.set_title("Audio features 3D por PALABRA " + ("(normalizado)" if use_norm else "(crudo)"))
ax.legend(loc="best", title="Palabra")
ax.grid(True)

plt.tight_layout()
plt.savefig(args.save, dpi=150)
print(f"Figura guardada en: {args.save}")

try:
    plt.show()
except Exception as e:
    print(f"No se pudo abrir ventana gráfica ({e}). Se guardó el PNG.")
