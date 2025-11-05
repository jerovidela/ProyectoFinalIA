import os
import json
import glob
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from collections import Counter

PARAM_KEYS = [
    "circle_area_ratio",
    "hu_moment_1",
    "angles_min",
    "hu_moment_2",
    "curvature_max",
]

n_clusters = 4
n_init = 10
max_iterations = 300
tolerance = 1e-4
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)
rng = np.random.RandomState(random_seed)

json_dir = "outjson"
pattern = os.path.join(json_dir, "*.json")
files = sorted(glob.glob(pattern))

filenames = []
labels = []
params_list = []

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        continue

    if "filename" in obj:
        filename = obj["filename"]
    else:
        filename = os.path.basename(fp)

    base = os.path.basename(filename)
    base_no_ext = os.path.splitext(base)[0]
    label = "".join(ch for ch in base_no_ext if not ch.isdigit())
    label = label.strip('_-. ').capitalize()

    params = []
    valid = True
    for k in PARAM_KEYS:
        if k in obj and obj[k] is not None:
            try:
                params.append(float(obj[k]))
            except Exception:
                valid = False
                break
        else:
            valid = False
            break

    if valid:
        filenames.append(filename)
        labels.append(label)
        params_list.append(params)

data_matrix = np.array(params_list, dtype=float)
n_samples, n_features = data_matrix.shape

scaler = StandardScaler()
X = scaler.fit_transform(data_matrix)

best_inertia = np.inf
best_centroids = None
best_labels = None

for init_run in range(n_init):
    
    centroids = np.empty((n_clusters, n_features), dtype=float)
    idx0 = rng.randint(n_samples)
    centroids[0] = X[idx0]

    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)

    for c in range(1, n_clusters):
        total = closest_sq.sum()
        if total == 0.0:
            idx = rng.randint(n_samples)
            centroids[c] = X[idx]
        else:
            probs = closest_sq / total
            r = rng.rand()
            idx = np.searchsorted(np.cumsum(probs), r)
            if idx >= n_samples:
                idx = n_samples - 1
            centroids[c] = X[idx]

        new_sq = np.sum((X - centroids[c]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    for iteration in range(max_iterations):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels_idx = d2.argmin(axis=1)

        for j in range(n_clusters):
            if not np.any(labels_idx == j):
                hardest = np.argmax(d2.min(axis=1))
                centroids[j] = X[hardest]
                labels_idx[hardest] = j

        new_centroids = np.zeros_like(centroids)
        for j in range(n_clusters):
            pts = X[labels_idx == j]
            if pts.size == 0:
                new_centroids[j] = X[rng.randint(n_samples)]
            else:
                new_centroids[j] = pts.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        base = max(1.0, np.linalg.norm(centroids))
        centroids = new_centroids
        if shift <= tolerance * base:
            break

    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    inertia = d2.min(axis=1).sum()

    if inertia < best_inertia:
        best_inertia = float(inertia)
        best_centroids = centroids.copy()
        best_labels = d2.argmin(axis=1).copy()

centroids = best_centroids
cluster_assignments = best_labels

print("COMPOSICIÓN DE CLUSTERS")
print("="*50)

for cluster_id in range(n_clusters):
    mask = cluster_assignments == cluster_id
    cluster_files = [filenames[i] for i in range(len(filenames)) if mask[i]]
    cluster_labels_list = [labels[i] for i in range(len(labels)) if mask[i]]

    print(f"\nCLUSTER {cluster_id} ({len(cluster_files)} elementos):")
    print("-" * 30)

    for i, (file, label) in enumerate(zip(cluster_files, cluster_labels_list)):
        print(f"  {i+1:2d}. {file} ({label})")

    type_counts = Counter(cluster_labels_list)
    print(f"\nResumen por tipo:")
    for tipo, count in sorted(type_counts.items()):
        percentage = (count / max(1, len(cluster_files))) * 100
        print(f"    {tipo}: {count} elementos ({percentage:.1f}%)")
