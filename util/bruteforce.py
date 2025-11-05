#!/usr/bin/env python3
"""
Brute-force search (exhaustive in given limits) for KMeans configurations
that produce 100% purity.

This version DOES NOT stop at the first success: it collects ALL configurations
that yield 100% purity and writes them to bruteforce_all_results.json.

Tweak:
- TOP_K_FEATURES: how many top-ranked features to consider
- MIN_SUBSET_SIZE / MAX_SUBSET_SIZE: subset sizes to try
- VERBOSE_EVERY: progress print frequency
"""
import json
import itertools
from pathlib import Path
from collections import defaultdict, Counter
import time

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

JSON_DIR = Path("outjson")
OUTPUT_FILE = Path("bruteforce_all_results.json")
N_CLUSTERS = 4
RANDOM_STATE = 42

# ---- tune these for runtime vs coverage ----
TOP_K_FEATURES = 18          # number of candidate features (from ranked list)
MIN_SUBSET_SIZE = 3          # min subset size to try
MAX_SUBSET_SIZE = 6          # max subset size to try (increase with caution)
VERBOSE_EVERY = 500          # print progress every N combos
# --------------------------------------------


def load_jsons(json_dir: Path):
    data = []
    names = []
    for p in sorted(json_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        data.append(j)
        names.append(p.name)
    return data, names


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def build_full_matrix(json_list):
    all_keys = set()
    for j in json_list:
        all_keys.update(j.keys())
    drop = {"filename", "label_hint"}

    feature_names = []
    for k in sorted(all_keys):
        if k in drop:
            continue
        sample_val = None
        for j in json_list:
            if k in j:
                sample_val = j[k]
                break
        if is_number(sample_val) or sample_val is None:
            feature_names.append(k)

    X = []
    y = []
    for j in json_list:
        row = []
        for k in feature_names:
            v = j.get(k, 0.0)
            if v is None:
                v = 0.0
            row.append(float(v))
        X.append(row)
        y.append(j.get("label_hint", "unknown"))
    X = np.array(X, dtype=float)
    y = np.array(y)
    return X, y, feature_names


def purity_score(y_true, y_pred):
    by_cluster = defaultdict(list)
    for yt, yp in zip(y_true, y_pred):
        by_cluster[yp].append(yt)
    correct = 0
    total = len(y_true)
    for _, arr in by_cluster.items():
        cnt = Counter(arr)
        correct += cnt.most_common(1)[0][1]
    return correct / total


def rank_features_by_separability(X, y, feature_names):
    classes = ["arandela", "tuerca", "tornillo", "clavo"]
    ranked = []
    for i, fname in enumerate(feature_names):
        vals = X[:, i]
        means = []
        stds = []
        for c in classes:
            mask = (y == c)
            if mask.any():
                v = vals[mask]
                means.append(v.mean())
                stds.append(v.std())
        if not means:
            score = 0.0
        else:
            score = (max(means) - min(means)) / (np.mean(stds) + 1e-6)
        ranked.append((score, i, fname))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


def build_class_means(X, y, classes, weights=None):
    cents = []
    for c in classes:
        mask = (y == c)
        if mask.any():
            cen = X[mask].mean(axis=0)
        else:
            cen = np.zeros(X.shape[1], dtype=float)
        if weights is not None:
            cen = cen * weights
        cents.append(cen)
    return np.vstack(cents)


def make_weights(cols, pattern):
    # cols: list of actual feature names (strings)
    w = np.ones(len(cols), dtype=float)
    if pattern is None:
        return w
    for i, fname in enumerate(cols):
        if pattern == "boost_angles" and "angle" in fname:
            w[i] = 2.5
        elif pattern == "boost_poly" and "polygon_vertices" in fname:
            w[i] = 3.0
        elif pattern == "boost_fft" and "contour_fft" in fname:
            w[i] = 3.0
        elif pattern == "boost_round" and ("circularity" in fname or "circle_area_ratio" in fname):
            w[i] = 2.5
    return w


def main():
    start_time = time.time()
    json_list, names = load_jsons(JSON_DIR)
    if not json_list:
        print("❌ No JSON files")
        return

    X, y, feature_names = build_full_matrix(json_list)
    # keep only labeled
    mask_known = np.isin(y, ["arandela", "tuerca", "tornillo", "clavo"])
    X = X[mask_known]
    y = y[mask_known]
    print(f"Loaded {X.shape[0]} labeled samples, {X.shape[1]} numeric features")

    # rank features
    ranked = rank_features_by_separability(X, y, feature_names)
    top_ranked = ranked[:TOP_K_FEATURES]
    top_idx = [idx for _, idx, _ in top_ranked]
    top_names = [feature_names[i] for i in top_idx]

    print("\nCandidate features (top):")
    for score, idx, fname in top_ranked:
        print(f"  {fname:35s}  score={score:6.2f}")

    # norm strategies
    norm_strategies = {
        "none": lambda A: A,
        "standard": lambda A: StandardScaler().fit_transform(A),
        "minmax": lambda A: MinMaxScaler().fit_transform(A),
    }

    weight_patterns = [None, "boost_angles", "boost_poly", "boost_fft", "boost_round"]

    classes = ["arandela", "tuerca", "tornillo", "clavo"]

    results = []   # collect all successful configs
    best_purity = 0.0
    total_checked = 0

    # iterate over subset sizes
    for k in range(MIN_SUBSET_SIZE, MAX_SUBSET_SIZE + 1):
        combos = list(itertools.combinations(range(TOP_K_FEATURES), k))
        print(f"\n=== Trying subset size k={k} → {len(combos)} combinations ===")
        for ci, comb in enumerate(combos, 1):
            feat_global_idx = [top_idx[i] for i in comb]
            feat_names = [feature_names[g] for g in feat_global_idx]
            Xsub = X[:, feat_global_idx]

            for norm_name, norm_fn in norm_strategies.items():
                Xn = norm_fn(Xsub)

                for wp in weight_patterns:
                    w = make_weights(feat_names, wp)
                    Xw = Xn * w

                    # init cents = class means in normalized space (and weighted)
                    init_cents = build_class_means(Xn, y, classes, weights=w)

                    km = KMeans(
                        n_clusters=N_CLUSTERS,
                        init=init_cents,
                        n_init=1,
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    )
                    y_pred = km.fit_predict(Xw)
                    purity = purity_score(y, y_pred)
                    total_checked += 1

                    if purity > best_purity:
                        best_purity = purity

                    if purity >= 1.0 - 1e-12:
                        # record success
                        # also record cluster -> class mapping
                        clusters = defaultdict(list)
                        for true_lbl, pred_lbl in zip(y, y_pred):
                            clusters[pred_lbl].append(true_lbl)
                        cluster_map = {int(c): dict(Counter(arr)) for c, arr in clusters.items()}

                        result = {
                            "subset_size": k,
                            "features": feat_names,
                            "norm": norm_name,
                            "weight_pattern": wp,
                            "purity": float(purity),
                            "cluster_map": cluster_map
                        }
                        results.append(result)

            # progress print
            if ci % VERBOSE_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"  ... {ci}/{len(combos)} combos tried (total checked {total_checked}) - best purity so far = {best_purity:.4f}  elapsed={elapsed:.1f}s")

    # end loops
    elapsed_total = time.time() - start_time
    print("\n=== SEARCH COMPLETE ===")
    print(f"Total combos checked: {total_checked}")
    print(f"Best purity seen: {best_purity:.4f}")
    print(f"Elapsed time: {elapsed_total:.1f}s")
    print(f"Total successful configurations (purity==1.0): {len(results)}")

    # save results
    out = {
        "summary": {
            "total_checked": total_checked,
            "best_purity": best_purity,
            "elapsed_seconds": elapsed_total,
            "top_k_features_considered": TOP_K_FEATURES,
            "min_subset_size": MIN_SUBSET_SIZE,
            "max_subset_size": MAX_SUBSET_SIZE,
        },
        "results": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved all results to: {OUTPUT_FILE}")

    # print a short list to console
    if results:
        print("\n=== Successful configs (first 20) ===")
        for r in results[:20]:
            print(f"- k={r['subset_size']} norm={r['norm']} wp={r['weight_pattern'] or 'none'} features={r['features']}")
    else:
        print("No 100% purity configuration found within the search limits.")

if __name__ == "__main__":
    main()
