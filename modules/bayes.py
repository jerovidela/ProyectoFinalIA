# bayes_module.py
# -*- coding: utf-8 -*-
"""
Reusable Bayesian estimator for the 'boxes' (a–d) model.

Public API:
    proporcion(labels: list[str]) -> dict[str, float]
        Returns expected class proportions (0..1) using Bayesian model averaging.
    contar(labels: list[str], total: int = 1000) -> dict[str, int]
        Returns expected counts over `total` items (default 1000), from the same proportions.

Labels must be any of:
    "tornillos", "clavos", "arandelas", "tuercas"
in any order and multiplicity (typically 10 items from your KNN output).
"""

import math
from collections import Counter
from typing import List, Dict

# Fixed class order
CLASSES = ("tornillo", "clavo", "arandela", "tuerca")

# Models (boxes) as proportions (sum to 1.0)
# a) 250/250/250/250
# b) 150/300/300/250
# c) 250/350/250/150
# d) 500/500/0/0
_MODELS = {
    "a": [250, 250, 250, 250],
    "b": [150, 300, 300, 250],
    "c": [250, 350, 250, 150],
    "d": [500, 500,   0,   0],
}
# Normalize to probabilities
for k in _MODELS:
    s = float(sum(_MODELS[k]))
    _MODELS[k] = [v / s for v in _MODELS[k]]

# --- Internal helpers (not exported) -----------------------------------------

def _counts_from_labels(labels: List[str]) -> List[int]:
    if not labels:
        raise ValueError("labels list is empty.")
    unknown = set(labels) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    c = Counter(labels)
    return [int(c.get(cls, 0)) for cls in CLASSES]

def _posterior_from_counts(counts: List[int]) -> Dict[str, float]:
    # Uniform prior over models
    prior_val = 1.0 / len(_MODELS)
    loglikes = {}
    for name, p in _MODELS.items():
        ll = 0.0
        impossible = False
        for c, prob in zip(counts, p):
            if prob == 0.0:
                if c > 0:
                    impossible = True
                    break
                # c == 0 contributes 0
            else:
                ll += c * math.log(prob)
        loglikes[name] = (-1e300 if impossible else ll)

    # Stable softmax with uniform prior
    m = max(loglikes.values())
    num = {k: prior_val * math.exp(loglikes[k] - m) for k in _MODELS.keys()}
    Z = sum(num.values())
    if Z == 0.0:
        # All impossible (shouldn't happen if counts consistent with at least one model)
        return {k: 0.0 for k in _MODELS.keys()}
    return {k: num[k] / Z for k in _MODELS.keys()}

def _expected_proportions(posterior: Dict[str, float]) -> List[float]:
    # E[p] = sum_k posterior(k) * p_k
    out = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        for name, p in _MODELS.items():
            out[i] += posterior[name] * p[i]
    return out

# --- Public API --------------------------------------------------------------

def proporcion(labels: List[str]) -> Dict[str, float]:
    """
    Returns expected proportions per class (0..1) as Bayesian model average.
    """
    counts = _counts_from_labels(labels)
    post = _posterior_from_counts(counts)
    exp_props = _expected_proportions(post)
    return {cls: prop for cls, prop in zip(CLASSES, exp_props)}

def contar(labels: List[str], total: int = 1000) -> Dict[str, int]:
    """
    Returns expected counts over `total` items using the same Bayesian average.
    """
    props = proporcion(labels)
    # Round to nearest integer while keeping sum close to `total`
    raw = {k: props[k] * total for k in props}
    rounded = {k: int(round(v)) for k, v in raw.items()}
    # Optional tiny adjustment to match total (rare off-by-one due to rounding)
    diff = total - sum(rounded.values())
    if diff != 0:
        # Distribute the difference by fractional parts, descending
        fracs = sorted(raw.items(), key=lambda kv: (kv[1] - math.floor(kv[1])), reverse=True)
        i = 0
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            k = fracs[i % len(fracs)][0]
            rounded[k] += step
            i += 1
    return rounded

# Optional: quick CLI check
if __name__ == "__main__":
    # Example: simulate 10 labels from your KNN output
    example = ["tornillos","tornillos","arandelas","tuercas","clavos",
               "arandelas","tornillos","clavos","arandelas","clavos"]
    print("Proporciones esperadas:", proporcion(example))
    print("Conteos esperados (1000):", contar(example, total=1000))
