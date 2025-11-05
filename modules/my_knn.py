#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, re
import numpy as np

def knn_from_featurejson(
    json_dir="features_json",
    feature_keys=("zcr_std_z", "rolloff95_std_z", "mfcc_std_4_z"),
    is_classification=True,
    n_neighbors=5,
    weights="distance",         # 'uniform' | 'distance'
    metric="manhattan",         # 'euclidean' | 'manhattan' | 'minkowski'
    p=2,                        # parámetro de minkowski
    standardize=True,
    test_fraction=0.3,
    random_seed=42,
    max_examples=10,
    verbose=True
):
    """
    KNN sobre JSONs con claves de features nombradas (no por índice).
    Usa SOLO: mfcc_std_4, zcr_std, rolloff95_std (por defecto).

    Returns (dict) con datos y resultados (accuracy o rmse).
    """
    # --- Cargar archivos ---
    pattern = os.path.join(json_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No se encontraron JSON en {json_dir}")

    names, X_list, y_list = [], [], []
    missing_counter = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        # Verificar que existan las 3 claves requeridas
        if not all(k in obj for k in feature_keys):
            missing_counter += 1
            continue

        # Extraer vector de 3 features
        try:
            vec = [float(obj[k]) for k in feature_keys]
        except Exception:
            continue

        # Sanitizar NaN/inf
        vec = np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()

        # Label
        label = obj.get("label", obj.get("class", obj.get("y")))
        if label is None:
            base = os.path.splitext(os.path.basename(fp))[0]
            m = re.match(r"([A-Za-zÁÉÍÓÚáéíóúñÑ]+)", base)
            label = m.group(1) if m else "unknown"

        X_list.append(vec)
        y_list.append(label)
        names.append(os.path.basename(fp))

    if verbose and missing_counter:
        print(f"Aviso: {missing_counter} archivos sin {list(feature_keys)} fueron ignorados.")

    X_all = np.asarray(X_list, dtype=float)
    y_raw = np.array(y_list, dtype=object)

    if X_all.size == 0:
        raise SystemExit("No se pudieron construir muestras válidas (X vacío).")

    n_samples, n_features = X_all.shape
    if n_features != len(feature_keys):
        raise SystemExit(f"Se esperaban {len(feature_keys)} features, llegaron {n_features}.")

    # y como enteros si clasificación
    class_to_int, int_to_class = None, None
    if is_classification:
        classes = np.unique(y_raw)
        class_to_int = {c: i for i, c in enumerate(classes)}
        int_to_class = {i: c for c, i in class_to_int.items()}
        y_all = np.array([class_to_int[c] for c in y_raw], dtype=int)
    else:
        try:
            y_all = y_raw.astype(float)
        except Exception:
            raise SystemExit("Para regresión, 'label'/'y' debe ser numérico o convertible a float.")

    # Split reproducible
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(n_samples)
    cut = int(round(n_samples * (1 - test_fraction)))
    tr, te = idx[:cut], idx[cut:]
    if len(te) == 0 or len(tr) == 0:
        raise SystemExit("Split inválido: ajusta test_fraction o agrega más datos.")

    X_train, y_train = X_all[tr], y_all[tr]
    X_test,  y_test  = X_all[te], y_all[te]
    names_test = [names[i] for i in te]

    if verbose:
        print(f"Split -> train={len(tr)} | test={len(te)}")

    # Estandarizar con stats de train (z-score)
    if standardize:
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        X_train = (X_train - mu) / sigma
        X_test  = (X_test  - mu) / sigma
    else:
        mu = np.zeros((1, n_features))
        sigma = np.ones((1, n_features))

    # Distancias (test vs train)
    A = X_test[:, None, :]   # (n_test, 1, d)
    B = X_train[None, :, :]  # (1, n_train, d)

    if metric == "euclidean":
        D = np.sqrt(((A - B) ** 2).sum(axis=2))
    elif metric == "manhattan":
        D = np.abs(A - B).sum(axis=2)
    elif metric == "minkowski":
        D = (np.abs(A - B) ** p).sum(axis=2) ** (1.0 / p)
    else:
        raise SystemExit("METRIC debe ser 'euclidean', 'manhattan' o 'minkowski'.")

    # K vecinos
    k = int(min(n_neighbors, len(X_train)))
    idx_sorted = np.argsort(D, axis=1)
    knn_idx   = idx_sorted[:, :k]               # (n_test, k)
    knn_dist  = np.take_along_axis(D, knn_idx, axis=1)
    knn_y     = y_train[knn_idx]

    # Pesos
    if weights == "uniform":
        W = np.ones_like(knn_dist, dtype=float)
    elif weights == "distance":
        W = 1.0 / np.maximum(knn_dist, 1e-12)
        # Si hay distancia 0 => pesos one-hot en esos vecinos
        has_inf = np.isinf(W)
        if has_inf.any():
            rows_with_inf = has_inf.any(axis=1)
            W_sub = W[rows_with_inf]
            mask_inf = has_inf[rows_with_inf]
            W_sub[~mask_inf] = 0.0
            W_sub[mask_inf] = 1.0
    else:
        raise SystemExit("WEIGHTS debe ser 'uniform' o 'distance'.")

    # Predicción
    if is_classification:
        n_classes = len(np.unique(y_train))
        scores = np.zeros((knn_y.shape[0], n_classes), dtype=float)
        for i in range(knn_y.shape[0]):
            scores[i] = np.bincount(knn_y[i], weights=W[i], minlength=n_classes)
        y_pred = scores.argmax(axis=1).astype(int)
        acc = (y_pred == y_test).mean() if len(y_test) else np.nan

        if verbose:
            print(f"\nAccuracy test: {acc:.4f}")
            print("\nEjemplos:")
            for i in range(min(max_examples, len(y_pred))):
                yt = int_to_class[int(y_test[i])]
                yp = int_to_class[int(y_pred[i])]
                print(f"  {names_test[i]:30s}  true={yt:10s}  pred={yp:10s}")

        metric_value = acc
    else:
        wsum = W.sum(axis=1, keepdims=True)
        wsum[wsum == 0] = 1.0
        y_pred = (W * knn_y).sum(axis=1) / wsum.ravel()
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2)) if len(y_test) else np.nan

        if verbose:
            print(f"\nRMSE test: {rmse:.6f}")
            print("\nEjemplos:")
            for i in range(min(max_examples, len(y_pred))):
                print(f"  {names_test[i]:30s}  true={float(y_test[i]):10.4f}  pred={float(y_pred[i]):10.4f}")

        metric_value = rmse

    if verbose:
        print("\nConfig usada:")
        print(f"  JSON_DIR      = {json_dir}")
        print(f"  FEATURE_KEYS  = {list(feature_keys)}")
        print(f"  IS_CLASSIF    = {is_classification}")
        print(f"  N_NEIGHBORS   = {n_neighbors}")
        print(f"  WEIGHTS       = {weights}")
        print(f"  METRIC        = {metric}")
        print(f"  P             = {p}")
        print(f"  STANDARDIZE   = {standardize}")
        print(f"  TEST_FRACTION = {test_fraction}")
        print(f"  RANDOM_SEED   = {random_seed}")

    return {
        "config": {
            "json_dir": json_dir,
            "feature_keys": list(feature_keys),
            "is_classification": is_classification,
            "n_neighbors": n_neighbors,
            "weights": weights,
            "metric": metric,
            "p": p,
            "standardize": standardize,
            "test_fraction": test_fraction,
            "random_seed": random_seed,
        },
        "filenames": names,
        "names_test": names_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metric_value": metric_value,
        "class_to_int": class_to_int,
        "int_to_class": int_to_class,
        "mu": mu,
        "sigma": sigma,
    }

# Ejemplo de uso:
# res = knn_from_featurejson(
#     json_dir="features_json",
#     feature_keys=("mfcc_std_4", "zcr_std", "rolloff95_std"),
#     is_classification=True,
#     n_neighbors=5,
#     weights="distance",
#     metric="manhattan",
#     p=2,
#     standardize=True,
#     test_fraction=0.3,
#     random_seed=42,
#     max_examples=10,
#     verbose=True,
# )
# print("Accuracy:", res["metric_value"])
