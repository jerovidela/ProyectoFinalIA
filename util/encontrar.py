#!/usr/bin/env python3
"""
Select the N images closest to their class centroid (per class).

- Reads one JSON per image from --data (default: outjson).
- Expected numeric keys: circle_area_ratio, hu_moment_1, hu_moment_2, angles_min, curvature_max
- Class label is taken from "label" if present; otherwise inferred from filename prefix.
- Standardizes features globally (StandardScaler) to avoid scale bias.
- For each class, computes centroid and picks the N closest samples.

Usage:
  python3 closest_to_centroid.py --data outjson --top 13
"""

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_KEYS = [
    "circle_area_ratio",
    "hu_moment_1",
    "hu_moment_2",
    "angles_min",
    "curvature_max",
]

def infer_label_from_name(name: str) -> str:
    base = Path(name).stem
    i = 0
    while i < len(base) and not base[i].isalpha():
        i += 1
    j = i
    while j < len(base) and base[j].isalpha():
        j += 1
    return base[i:j] if i < j else base

def load_dataset(data_dir: Path):
    X, y, names = [], [], []
    skipped = []
    for p in sorted(data_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            skipped.append((p.name, "invalid_json"))
            continue

        feats = []
        ok = True
        for k in FEATURE_KEYS:
            v = j.get(k, None)
            if isinstance(v, (int, float)):
                feats.append(float(v))
            else:
                ok = False
                break
        if not ok:
            skipped.append((p.name, "missing_feature"))
            continue

        label = j.get("label")
        if not label:
            label = infer_label_from_name(j.get("filename", p.name))

        X.append(feats)
        y.append(label)
        names.append(j.get("filename", p.name))
    return np.array(X, dtype=float), np.array(y, dtype=object), np.array(names, dtype=object), skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="outjson", help="Folder with JSON feature files")
    ap.add_argument("--top", type=int, default=13, help="Top-N closest per class")
    args = ap.parse_args()

    data_dir = Path(args.data)
    X, y, names, skipped = load_dataset(data_dir)
    if X.size == 0:
        print("No valid samples found. Check JSON structure and feature keys.")
        return
    if skipped:
        print(f"Skipped {len(skipped)} files (missing features / invalid JSON).")

    # Global standardization
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Compute per-class centroid and distances
    rows = []
    classes = sorted(set(y))
    for cls in classes:
        idx = np.where(y == cls)[0]
        Xc = Xs[idx]
        centroid = Xc.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        order = np.argsort(dists)
        top_n = order[: min(args.top, len(order))]
        for rank, k in enumerate(top_n, start=1):
            rows.append({
                "label": cls,
                "filename": names[idx[k]],
                "distance_to_centroid": float(dists[k]),
                "rank_within_class": rank
            })

    df = pd.DataFrame(rows).sort_values(["label", "rank_within_class"])
    df.to_csv("closest_to_centroid.csv", index=False)

    print(f"Done. Wrote closest_to_centroid.csv with the top {args.top} closest per class.")
    print("Columns: label, filename, distance_to_centroid, rank_within_class")

if __name__ == "__main__":
    main()
