import json
import os
import glob
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Any

EPS = 1e-8

def compute_features_exact(path: str) -> Tuple[float, float, float]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    peak = np.max(np.abs(y))
    y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_std = mfcc.std(axis=1) + EPS
    mfcc_std_4 = float(mfcc_std[3])

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))**2
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)
    rolloff95_std = float(roll.std() + EPS)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_std = float(zcr.std() + EPS)

    return zcr_std, rolloff95_std, mfcc_std_4


def extract_and_save_features(
    in_dir: str = "audio",
    out_dir: str = "features_json",
    stats_dir: str = "features_json_data",
    pattern: str = "*.wav"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    stats_path = Path(stats_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stats_path.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(str(in_path), pattern)))
    if not files:
        print(f"No se encontraron WAV en {in_path}")
        return [], {}

    records: List[Dict[str, Any]] = []
    for fp in files:
        zcr_std, rolloff95_std, mfcc_std_4 = compute_features_exact(fp)
        records.append({
            "path": fp,
            "zcr_std": zcr_std,
            "rolloff95_std": rolloff95_std,
            "mfcc_std_4": mfcc_std_4,
        })

    zcr_vals   = np.array([r["zcr_std"] for r in records], dtype=float)
    roll_vals  = np.array([r["rolloff95_std"] for r in records], dtype=float)
    mfcc4_vals = np.array([r["mfcc_std_4"] for r in records], dtype=float)

    zcr_mu  = float(np.nanmean(zcr_vals))
    zcr_sd  = float(np.nanstd(zcr_vals) + EPS)
    roll_mu = float(np.nanmean(roll_vals))
    roll_sd = float(np.nanstd(roll_vals) + EPS)
    mf4_mu  = float(np.nanmean(mfcc4_vals))
    mf4_sd  = float(np.nanstd(mfcc4_vals) + EPS)

    stats = {
        "z_score_stats": {
            "zcr_std":       {"mean": zcr_mu, "std": zcr_sd},
            "rolloff95_std": {"mean": roll_mu, "std": roll_sd},
            "mfcc_std_4":    {"mean": mf4_mu, "std": mf4_sd},
        },
        "count": len(records)
    }

    for r in records:
        z_zcr   = float((r["zcr_std"]       - zcr_mu) / zcr_sd)  if np.isfinite(r["zcr_std"])       else np.nan
        z_roll  = float((r["rolloff95_std"] - roll_mu) / roll_sd) if np.isfinite(r["rolloff95_std"]) else np.nan
        z_mfcc4 = float((r["mfcc_std_4"]    - mf4_mu) / mf4_sd)  if np.isfinite(r["mfcc_std_4"])    else np.nan

        out_obj = {
            "path": r["path"],
            "zcr_std": r["zcr_std"],
            "rolloff95_std": r["rolloff95_std"],
            "mfcc_std_4": r["mfcc_std_4"],
            "zcr_std_z": z_zcr,
            "rolloff95_std_z": z_roll,
            "mfcc_std_4_z": z_mfcc4
        }

        base = os.path.splitext(os.path.basename(r["path"]))[0]
        out_fp = out_path / f"{base}.json"
        out_fp.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    (stats_path / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK. Generados {len(records)} JSON en: {out_path.resolve()}")
    print(f"Stats del dataset guardadas en: {(stats_path / 'dataset_stats.json').resolve()}")

    return records, stats

