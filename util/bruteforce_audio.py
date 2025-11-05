#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install numpy scikit-learn

import os, re, glob, json, itertools, csv, sys, time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif

# ------------------ CONFIG ------------------
DATA_DIR = "features_json"    # carpeta con .json
SUBSET_SIZES = [3, 4, 5]      # tamaños de subconjunto
TOPK_POOL = 18                # EXACTO: primero elegimos los 18 mejores por ANOVA-F
USE_LOSO = False              # True: evalúa Leave-One-Speaker-Out (más estricto)
N_SPLITS = 5                  # si USE_LOSO=False, k-fold estratificado
SEED = 42

# KNN fijo (valores que te funcionaron bien)
KNN_N = 5
KNN_METRIC = "manhattan"      # "euclidean" o "manhattan"
KNN_WEIGHTS = "distance"      # "uniform" o "distance"

# Salidas
CSV_PATH = "feature_search_results.csv"
TOP18_JSON = "top18_features.json"
PERFECT_JSON = "perfect_feature_sets.json"
PRINT_TOP_N = 15              # imprime top-N por k en consola
ACC_PERFECT_EPS = 1e-4        # tolerancia para considerar 100%
# -------------------------------------------

np.random.seed(SEED)

# --- Cargar JSONs ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
if len(files) == 0:
    print(f"⚠️  No se encontraron JSON en {DATA_DIR}")
    sys.exit(1)

X_list, y_list, names_list, spk_list = [], [], [], []
feature_names = None

RE_WORD = re.compile(r"^(?P<word>[A-Za-zÁÉÍÓÚáéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)
RE_SPK  = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúñÑ]+_(?P<spk>[A-Za-z0-9ÁÉÍÓÚáéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    vec = np.asarray(obj["feature_vector"], dtype=float)
    X_list.append(vec)

    fname = obj.get("filename", os.path.basename(fp).replace(".json", ".wav"))
    label = obj.get("label", None)
    if label is None:
        m = RE_WORD.match(os.path.basename(fname))
        if not m:
            raise ValueError(f"No puedo extraer la palabra de: {fname}")
        label = m.group("word").lower()

    spk = obj.get("speaker", None)
    if spk is None:
        m2 = RE_SPK.match(os.path.basename(fname))
        spk = m2.group("spk").lower() if m2 else "unknown"

    y_list.append(label)
    names_list.append(fname)
    spk_list.append(spk)

    if feature_names is None:
        feature_names = obj.get("feature_names", None)

X = np.vstack(X_list)
y = np.array(y_list)
names = np.array(names_list)
groups = np.array(spk_list)

classes = sorted(set(y.tolist()))
print(f"Encontrados {len(files)} JSON | X={X.shape}, clases={classes}")

# Si no hay nombres de features, generamos genéricos
if feature_names is None:
    feature_names = [f"f{i}" for i in range(X.shape[1])]
elif len(feature_names) != X.shape[1]:
    raise ValueError("feature_names y feature_vector difieren en longitud.")

# --- Paso 1: Ranking univariante (ANOVA-F) para seleccionar TOP 18 ---
scaler_all = StandardScaler()
X_std = scaler_all.fit_transform(X)

F_vals, p_vals = f_classif(X_std, y)
F_vals = np.nan_to_num(F_vals, nan=0.0, posinf=0.0, neginf=0.0)

rank_idx = np.argsort(F_vals)[::-1]
pool_idx = rank_idx[:min(TOPK_POOL, X.shape[1])]

top18_data = {
    "topk": int(len(pool_idx)),
    "indices": [int(i) for i in pool_idx.tolist()],
    "names": [feature_names[i] for i in pool_idx],
    "F_scores": [float(F_vals[i]) for i in pool_idx]
}
with open(TOP18_JSON, "w", encoding="utf-8") as f:
    json.dump(top18_data, f, indent=2, ensure_ascii=False)

print(f"TOP {len(pool_idx)} por ANOVA-F guardados en: {TOP18_JSON}")
print("Top-10 preliminares:", [feature_names[i] for i in pool_idx[:10]])

# --- Preparar CV ---
if USE_LOSO:
    cv = LeaveOneGroupOut()
    cv_splits = list(cv.split(X, y, groups=groups))
    print(f"Evaluación: LOSO por orador | folds={len(cv_splits)} | oradores={sorted(set(groups.tolist()))}")
else:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_splits = list(cv.split(X, y))
    print(f"Evaluación: StratifiedKFold | folds={len(cv_splits)}")

# --- Paso 2: Fuerza bruta con el pool de 18 ---
all_rows = []
perfect_sets = {3: [], 4: [], 5: []}

for k in SUBSET_SIZES:
    combs = list(itertools.combinations(pool_idx, k))
    total = len(combs)
    print(f"\nCombinaciones para k={k} dentro de {len(pool_idx)} features -> {total} combinaciones")

    best_for_k = []
    t0 = time.time()
    checked = 0
    last_pct = -1

    for idx_tuple in combs:
        X_sub = X[:, idx_tuple]

        # CV con escalado dentro de cada fold
        scores = []
        for tr_idx, te_idx in cv_splits:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_sub[tr_idx])
            Xte = scaler.transform(X_sub[te_idx])

            clf = KNeighborsClassifier(n_neighbors=KNN_N, metric=KNN_METRIC, weights=KNN_WEIGHTS)
            clf.fit(Xtr, y[tr_idx])
            yhat = clf.predict(Xte)
            scores.append(accuracy_score(y[te_idx], yhat))

        acc = float(np.mean(scores))
        best_for_k.append((acc, idx_tuple))

        # guardar fila para CSV
        all_rows.append([
            k,
            f"{acc:.6f}",
            " ".join(map(str, idx_tuple)),
            "|".join(feature_names[i] for i in idx_tuple)
        ])

        # si es "perfecto", guardar en JSON
        if acc >= 1.0 - ACC_PERFECT_EPS:
            perfect_sets[k].append({
                "accuracy": acc,
                "indices": [int(i) for i in idx_tuple],
                "features": [feature_names[i] for i in idx_tuple]
            })

        # progreso
        checked += 1
        pct = int(checked * 100 / total)
        if pct >= last_pct + 5:
            last_pct = pct
            elapsed = time.time() - t0
            eta = (elapsed / checked) * (total - checked) if checked > 0 else 0
            print(f"  Progreso {pct:3d}% | mejor={max(best_for_k)[0]:.4f} | elapsed={elapsed:.1f}s | ETA~{eta:.1f}s")

    # imprimir top-N
    best_for_k.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {min(PRINT_TOP_N, len(best_for_k))} combinaciones para k={k}:")
    for rank, (sc, idx_tuple) in enumerate(best_for_k[:PRINT_TOP_N], 1):
        feats = [feature_names[i] for i in idx_tuple]
        print(f"  #{rank:02d} | acc={sc:.4f} | feats={feats}")

# --- Guardar CSV con todas las combinaciones evaluadas ---
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "accuracy", "features_idx", "features_names"])
    writer.writerows(all_rows)

print(f"\n✅ Resultados completos guardados en: {CSV_PATH}")

# --- Guardar JSON con sets "perfectos" (100%) ---
with open(PERFECT_JSON, "w", encoding="utf-8") as f:
    json.dump(perfect_sets, f, indent=2, ensure_ascii=False)

total_perfect = sum(len(perfect_sets[k]) for k in perfect_sets)
print(f"✅ Conjuntos con 100% de accuracy guardados en: {PERFECT_JSON} (total={total_perfect})")
print("   Sugerencia: para generalización real a voces nuevas, probá USE_LOSO=True.")
