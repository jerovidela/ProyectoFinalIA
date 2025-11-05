#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install scikit-learn numpy

import os, re, glob, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = "features_json"
FEATURES = [
        68,
        214,
        213,
        242,
        66
      ]
TEST_SIZE = 0.30
RANDOM_SEED = 42

# Grid de KNN a probar
PARAM_GRID = {
    "kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9],
    "kneighborsclassifier__metric": ["euclidean", "manhattan"],
    "kneighborsclassifier__weights": ["uniform", "distance"],
}
N_SPLITS_CV = 5
N_REPEATS_HOLDOUT = 5

FILE_RE_WORD = re.compile(r"^(?P<word>[A-Za-záéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)
FILE_RE_SPK  = re.compile(r"^[A-Za-záéíóúñÑ]+_(?P<spk>[A-Za-z0-9áéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
if not files:
    raise SystemExit(f"❌ No se encontraron archivos en {DATA_DIR}")
print(f"Encontrados {len(files)} archivos de features")

X, y, names, spk = [], [], [], []
for fp in files:
    obj = json.load(open(fp, "r", encoding="utf-8"))
    feats = obj["feature_vector"]
    vec = [float(feats[f]) for f in FEATURES]
    X.append(vec)

    fname = obj["filename"]
    names.append(fname)

    m = FILE_RE_WORD.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"No pude extraer la palabra de {fname}")
    y.append(m.group("word").lower())

    m2 = FILE_RE_SPK.match(os.path.basename(fname))
    spk.append(m2.group("spk").lower() if m2 else "unknown")

X = np.array(X, float)
y = np.array(y)
spk = np.array(spk)
classes = sorted(set(y.tolist()))

print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features, clases={classes}")
print(f"Oradores: {sorted(set(spk.tolist()))}")

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
grid = GridSearchCV(pipe, PARAM_GRID, cv=cv, n_jobs=-1)
grid.fit(X, y)
print("\n=== GRID SEARCH (CV interna) ===")
print("Mejores hiperparámetros:", grid.best_params_)
print(f"CV accuracy (media): {grid.best_score_:.4f}")

best_model = grid.best_estimator_

print("\n=== HOLD-OUT repetido (promedio) ===")
accs = []
for i, seed in enumerate([RANDOM_SEED + k*17 for k in range(N_REPEATS_HOLDOUT)], 1):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    best_model.fit(Xtr, ytr)
    yhat = best_model.predict(Xte)
    acc = accuracy_score(yte, yhat)
    accs.append(acc)
    print(f"  Split {i} | seed={seed} | acc={acc:.4f}")
print(f"Accuracy promedio (hold-out x{N_REPEATS_HOLDOUT}): {np.mean(accs):.4f} ± {np.std(accs):.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("\n=== RESULTADOS (split fijo) ===")
print(classification_report(y_test, y_pred, digits=4))
print("Matriz de confusión (orden mostrado):", classes)
print(confusion_matrix(y_test, y_pred, labels=classes))

print("\n=== LOSO por orador ===")
logo = LeaveOneGroupOut()
loso_accs = []
for tr, te in logo.split(X, y, groups=spk):
    best_model.fit(X[tr], y[tr])
    yhat = best_model.predict(X[te])
    acc = accuracy_score(y[te], yhat)
    loso_accs.append(acc)
    held = sorted(set(spk[te].tolist()))
    print(f"  Held-out {held} | acc={acc:.3f}")
print(f"Mean LOSO accuracy: {np.mean(loso_accs):.3f}")
