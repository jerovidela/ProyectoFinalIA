import os, json, glob, re
import numpy as np

JSON_DIR = "features_json"
FEATURES = [68, 214, 213, 242, 66]

IS_CLASSIFICATION = True
N_NEIGHBORS = 5
WEIGHTS = "distance"       # 'uniform' | 'distance'
METRIC = "manhattan"       # 'euclidean' | 'manhattan' | 'minkowski'
P = 2                      # parámetro de minkowski (si aplica)
STANDARDIZE = True         # estandarizar X con media/var de train
TEST_FRACTION = 0.3
RANDOM_SEED = 42

pattern = os.path.join(JSON_DIR, "*.json")
files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"No se encontraron JSON en {JSON_DIR}")

names = []
X_list = []
y_list = []

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    feats = obj["feature_vector"]
    vec = [float(feats[i]) for i in FEATURES]

    label = obj.get("label", obj.get("class", obj.get("y")))
    if label is None:
        base = os.path.splitext(os.path.basename(fp))[0]
        m = re.match(r"([A-Za-z]+)", base)
        label = m.group(1) if m else "unknown"

    X_list.append(vec)
    y_list.append(label)
    names.append(os.path.basename(fp))

X_all = np.asarray(X_list, dtype=float)
y_raw = np.array(y_list, dtype=object)
n_samples, n_features = X_all.shape

if IS_CLASSIFICATION:
    classes = np.unique(y_raw)
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_all = np.array([class_to_int[c] for c in y_raw], dtype=int)
else:
    try:
        y_all = y_raw.astype(float)
    except:
        raise SystemExit("Para regresión, 'label'/'y' debe ser numérico o convertible a float.")

rng = np.random.default_rng(RANDOM_SEED)
idx = rng.permutation(n_samples)
cut = int(round(n_samples * (1 - TEST_FRACTION)))
tr, te = idx[:cut], idx[cut:]
X_train, y_train = X_all[tr], y_all[tr]
X_test,  y_test  = X_all[te], y_all[te]
names_test = [names[i] for i in te]

print(f"Split -> train={len(tr)} | test={len(te)}")

if STANDARDIZE:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma
else:
    mu = np.zeros((1, n_features))
    sigma = np.ones((1, n_features))

A = X_test[:, None, :]
B = X_train[None, :, :]

if METRIC == "euclidean":
    D = np.sqrt(((A - B) ** 2).sum(axis=2))
elif METRIC == "manhattan":
    D = np.abs(A - B).sum(axis=2)
elif METRIC == "minkowski":
    D = (np.abs(A - B) ** P).sum(axis=2) ** (1.0 / P)
else:
    raise SystemExit("METRIC debe ser 'euclidean', 'manhattan' o 'minkowski'.")

idx_sorted = np.argsort(D, axis=1)
knn_idx   = idx_sorted[:, :N_NEIGHBORS]                 # (n_test, k)
knn_dist  = np.take_along_axis(D, knn_idx, axis=1)      # (n_test, k)
knn_y     = y_train[knn_idx]                            # (n_test, k)

if WEIGHTS == "uniform":
    W = np.ones_like(knn_dist, dtype=float)
elif WEIGHTS == "distance":
    W = np.empty_like(knn_dist, dtype=float)
    W[:] = 1.0 / np.maximum(knn_dist, 1e-12)
    W[knn_dist == 0.0] = np.inf
else:
    raise SystemExit("WEIGHTS debe ser 'uniform' o 'distance'.")

if WEIGHTS == "distance":
    has_inf = np.isinf(W)
    rows_with_inf = has_inf.any(axis=1)
    if rows_with_inf.any():
        W_sub = W[rows_with_inf]
        mask_inf = has_inf[rows_with_inf]
        W_sub[~mask_inf] = 0.0
        W_sub[mask_inf] = 1.0

if IS_CLASSIFICATION:
    n_classes = len(np.unique(y_train))
    scores = np.zeros((knn_y.shape[0], n_classes), dtype=float)
    for i in range(knn_y.shape[0]):
        scores[i] = np.bincount(knn_y[i], weights=W[i], minlength=n_classes)
    y_pred = scores.argmax(axis=1).astype(int)
    acc = (y_pred == y_test).mean() if len(y_test) else np.nan
    print(f"\nAccuracy test: {acc:.4f}")
    inv_map = {v: k for k, v in ({} if not IS_CLASSIFICATION else {c:i for i,c in enumerate(np.unique(y_raw))}.items())}
    inv_map = {v: k for k, v in ({c:i for i,c in enumerate(np.unique(y_raw))}).items()}
    print("\nEjemplos:")
    for i in range(min(10, len(y_pred))):
        yt = inv_map[int(y_test[i])]
        yp = inv_map[int(y_pred[i])]
        print(f"  {names_test[i]:30s}  true={yt:10s}  pred={yp:10s}")
else:
    wsum = W.sum(axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    y_pred = (W * knn_y).sum(axis=1) / wsum.ravel()
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2)) if len(y_test) else np.nan
    print(f"\nRMSE test: {rmse:.6f}")
    print("\nEjemplos:")
    for i in range(min(10, len(y_pred))):
        print(f"  {names_test[i]:30s}  true={float(y_test[i]):10.4f}  pred={float(y_pred[i]):10.4f}")

print("\nConfig usada:")
print(f"  JSON_DIR      = {JSON_DIR}")
print(f"  FEATURES      = {FEATURES}")
print(f"  IS_CLASSIF    = {IS_CLASSIFICATION}")
print(f"  N_NEIGHBORS   = {N_NEIGHBORS}")
print(f"  WEIGHTS       = {WEIGHTS}")
print(f"  METRIC        = {METRIC}")
print(f"  P             = {P}")
print(f"  STANDARDIZE   = {STANDARDIZE}")
print(f"  TEST_FRACTION = {TEST_FRACTION}")
print(f"  RANDOM_SEED   = {RANDOM_SEED}")
