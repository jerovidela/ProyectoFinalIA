#!/usr/bin/env python3
import os
import glob
import json
import random
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans  # intra-class clustering

# ---------------- CONFIG ----------------
JSON_DIR = "outjson"
PARAM_KEYS = [
    "circle_area_ratio",
    "hu_moment_1",
    "angles_min",
    "hu_moment_2",
    "curvature_max",
]

DESIRED_PER_CLASS = 18        # lo que vos querés
MIN_PER_CLASS = 10            # no bajar de 10
N_FINAL_CLUSTERS = 4
MAX_ITER_FINAL = 1000          # esto sí es razonable
TOL_FINAL = 1e-4
N_INIT_FINAL = 10
RANDOM_SEED = 42
CANDIDATES_PER_SIZE = 1000      # 50 intentos por tamaño es más que suficiente
# ----------------------------------------


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_dataset(json_dir):
    pattern = os.path.join(json_dir, "*.json")
    files = sorted(glob.glob(pattern))

    feats, labels, names = [], [], []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        filename = obj.get("filename", os.path.basename(fp))
        base = os.path.basename(filename)
        base_no_ext = os.path.splitext(base)[0]
        label = "".join(ch for ch in base_no_ext if not ch.isdigit())
        label = label.strip("_-. ").capitalize()

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

        if not valid:
            continue

        feats.append(params)
        labels.append(label)
        names.append(filename)

    if not feats:
        raise RuntimeError(f"No valid samples found in {json_dir}")

    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=object)
    names = np.array(names, dtype=object)
    return X, y, names


def kmeans_single_run(X, n_clusters, max_iter, tol, rng):
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features), dtype=float)

    # k-means++ init
    first_idx = rng.integers(0, n_samples)
    centroids[0] = X[first_idx]

    for c in range(1, n_clusters):
        dists = np.full(n_samples, np.inf)
        for i in range(n_samples):
            for j in range(c):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                if dist < dists[i]:
                    dists[i] = dist
        probs = dists / dists.sum()
        chosen = rng.choice(n_samples, p=probs)
        centroids[c] = X[chosen]

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # assign
        for i in range(n_samples):
            d = np.sum((centroids - X[i]) ** 2, axis=1)
            labels[i] = np.argmin(d)

        # update
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                new_centroids[k] = pts.mean(axis=0)
            else:
                new_centroids[k] = X[rng.integers(0, n_samples)]

        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids
        if shift < tol:
            break

    inertia = 0.0
    for i in range(n_samples):
        inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)

    return labels, centroids, inertia


def kmeans_best_of_n_init(X, n_clusters, max_iter, tol, n_init, rng):
    best_inertia = float("inf")
    best_labels = None
    best_centroids = None
    for _ in range(n_init):
        labels, centroids, inertia = kmeans_single_run(
            X, n_clusters, max_iter, tol, rng
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids, best_inertia


def purity_score(cluster_labels, true_labels, n_clusters):
    n = len(true_labels)
    total_correct = 0
    for k in range(n_clusters):
        mask = (cluster_labels == k)
        size = mask.sum()
        if size == 0:
            return 0.0, False
        labs = true_labels[mask]
        cnt = Counter(labs)
        total_correct += cnt.most_common(1)[0][1]
    return total_correct / n, True


def cluster_within_class(X_class, n_subclusters, random_state):
    n_samples = X_class.shape[0]
    n_clusters = min(n_subclusters, n_samples)
    if n_clusters <= 1:
        return [list(range(n_samples))]
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=random_state,
    )
    km.fit(X_class)
    labels = km.labels_
    centers = km.cluster_centers_

    buckets = []
    for k in range(n_clusters):
        idxs = np.where(labels == k)[0]
        center = centers[k]
        dists = np.linalg.norm(X_class[idxs] - center, axis=1)
        ordered = [idx for _, idx in sorted(zip(dists, idxs), key=lambda x: x[0])]
        buckets.append(ordered)
    return buckets


def build_candidate_from_buckets(label_buckets, labels_sorted, target_per_class, shuffle=True):
    candidate = {}
    for lab in labels_sorted:
        buckets = label_buckets[lab]["buckets"]
        order = list(range(len(buckets)))
        if shuffle:
            random.shuffle(order)
        picked = []
        ptrs = [0] * len(buckets)

        while len(picked) < target_per_class:
            did = False
            for bi in order:
                if len(picked) >= target_per_class:
                    break
                if ptrs[bi] < len(buckets[bi]):
                    picked.append(buckets[bi][ptrs[bi]])
                    ptrs[bi] += 1
                    did = True
            if not did:
                break
        candidate[lab] = picked
    return candidate


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    X, y, names = load_dataset(JSON_DIR)

    # group by label
    label_to_indices = defaultdict(list)
    for i, lab in enumerate(y):
        label_to_indices[lab].append(i)

    print("Label counts:")
    for lab, idxs in label_to_indices.items():
        print(f"  {lab}: {len(idxs)}")

    labels_sorted = sorted(label_to_indices.keys())
    min_available = min(len(idxs) for idxs in label_to_indices.values())

    # máximo que realmente podemos pedir (18 en tu caso, porque arandela=18)
    effective_start = min(DESIRED_PER_CLASS, min_available)
    print(f"\nWe will try from {effective_start} imgs/class downwards (because of availability).")

    # 1) construir TODOS los buckets por clase
    label_buckets = {}
    for lab in labels_sorted:
        idxs = label_to_indices[lab]
        X_class = X[idxs]

        # normalizar dentro de la clase antes de sklearn
        m = X_class.mean(axis=0)
        s = X_class.std(axis=0, ddof=0)
        s[s == 0.0] = 1.0
        Xc_norm = (X_class - m) / s

        buckets_local = cluster_within_class(
            Xc_norm,
            n_subclusters=effective_start,
            random_state=RANDOM_SEED
        )

        buckets_global = []
        for lst in buckets_local:
            buckets_global.append([idxs[x] for x in lst])

        label_buckets[lab] = {"buckets": buckets_global}

    # 2) ahora sí: PROBAR tamaños (fuera del for anterior!)
    start = effective_start
    stop = max(MIN_PER_CLASS, 1)
    print(f"\nWe will try from {start} imgs/class down to {stop} (not below {MIN_PER_CLASS}).")

    pure_solutions = []

    for target_per_class in range(start, stop - 1, -1):
        print(f"\n--- Trying {target_per_class} images per class ---")
        for attempt in range(CANDIDATES_PER_SIZE):
            candidate = build_candidate_from_buckets(
                label_buckets,
                labels_sorted,
                target_per_class,
                shuffle=True
            )

            # alguno no llegó -> no sirve
            if any(len(candidate[lab]) < target_per_class for lab in labels_sorted):
                continue

            subset_indices = []
            subset_labels = []
            for lab in labels_sorted:
                subset_indices.extend(candidate[lab])
                subset_labels.extend([lab] * target_per_class)

            subset_indices = np.array(subset_indices, dtype=int)
            subset_labels = np.array(subset_labels, dtype=object)

            X_sub = X[subset_indices]

            # normalización estándar global
            means = X_sub.mean(axis=0)
            stds = X_sub.std(axis=0, ddof=0)
            stds[stds == 0.0] = 1.0
            Xn = (X_sub - means) / stds

            km_labels, _, inertia = kmeans_best_of_n_init(
                Xn,
                N_FINAL_CLUSTERS,
                MAX_ITER_FINAL,
                TOL_FINAL,
                N_INIT_FINAL,
                rng
            )
            purity, no_empty = purity_score(km_labels, subset_labels, N_FINAL_CLUSTERS)

            if purity == 1.0 and no_empty:
                pure_solutions.append({
                    "per_class": target_per_class,
                    "indices": subset_indices,
                    "labels": subset_labels,
                    "kmeans_labels": km_labels,
                    "inertia": float(inertia),
                })
                print(f"  [OK] {target_per_class}/class attempt={attempt+1} -> 100% pure, inertia={inertia:.6f}")

        # si con este tamaño ya encontramos al menos 1, no seguimos bajando
        if any(sol["per_class"] == target_per_class for sol in pure_solutions):
            break

    if not pure_solutions:
        print("\nNo 100% pure dataset found.")
        return

    # ordenar: primero más imágenes por clase, luego menor inertia
    pure_solutions.sort(key=lambda s: (-s["per_class"], s["inertia"]))

    print("\n============================")
    print("PURE DATASETS FOUND")
    print("============================")
    for sol in pure_solutions:
        pc = sol["per_class"]
        print(f"\nDataset with {pc} per class (total={pc * len(labels_sorted)}) | inertia={sol['inertia']:.6f}")
        idxs = sol["indices"]
        labs = sol["labels"]
        kml = sol["kmeans_labels"]
        files = names[idxs]
        for k in range(N_FINAL_CLUSTERS):
            mask = (kml == k)
            print(f"  Cluster {k} ({mask.sum()}):")
            for f, lab in zip(files[mask], labs[mask]):
                print(f"    {f} ({lab})")


if __name__ == "__main__":
    main()
