#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Requisitos: pip install librosa numpy soundfile

import os, re, json
import numpy as np
import librosa
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_DIR = Path("audio")           # carpeta con .wav
OUTPUT_DIR = Path("features_json")  # salida de .json
SR = 16000                          # Hz
TOP_DB_TRIM = 40                    # recorte silencios
N_MELS = 40
N_MFCC = 13
LPC_ORDER = 16
# valores "objetivo" (se ajustarán dinámicamente si el audio es corto)
FRAME_LEN_TARGET = 400    # ~25 ms a 16k
HOP_TARGET = 160          # ~10 ms a 16k
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex para etiqueta y speaker: palabra_orador_XX.wav
RE_FULL = re.compile(r"^(?P<word>[A-Za-zÁÉÍÓÚáéíóúñÑ]+)_(?P<spk>[A-Za-z0-9ÁÉÍÓÚáéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)
RE_WORD = re.compile(r"^(?P<word>[A-Za-zÁÉÍÓÚáéíóúñÑ]+)_[^/\\]+\.wav$", re.IGNORECASE)
# ---------------------------------------

def _best_nfft(L):
    # Elige la mayor potencia "típica" que quepa en L
    for cand in (2048, 1024, 512, 256, 128, 64):
        if L >= cand:
            return cand
    return max(32, L)

files = sorted(INPUT_DIR.glob("*.wav"))
print(f"Encontrados {len(files)} audios")

for path in files:
    print(f"Procesando: {path.name}")

    # --------- carga / preproc básico ----------
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y is None or len(y) == 0:
        print("  ⚠️ Audio vacío, salto"); continue

    y, _ = librosa.effects.trim(y, top_db=TOP_DB_TRIM)
    if len(y) == 0:
        print("  ⚠️ Todo silencio tras trim, salto"); continue

    y = librosa.util.normalize(y)
    duration_s = float(len(y) / sr)  # duración real tras trim (antes de padding)

    # Robustez a señales cortas: pad mínimo
    MIN_SAMPLES = 1024
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))

    # --------- etiquetas desde filename ----------
    base = os.path.basename(path.name)
    m = RE_FULL.match(base)
    if m:
        label = m.group("word").lower()
        speaker = m.group("spk").lower()
    else:
        m2 = RE_WORD.match(base)
        if not m2:
            raise ValueError(f"No puedo extraer etiqueta de: {base}")
        label = m2.group("word").lower()
        speaker = None

    # --------- Parámetros dinámicos por archivo ----------
    frame_length = min(FRAME_LEN_TARGET, len(y))
    hop_length = max(1, frame_length // 4)
    n_fft = _best_nfft(len(y))

    # --------- Módulos de features ---------------
    feature_vector = []
    feature_names = []

    # --- Mel & MFCC (+ CMVN) ---
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=n_fft, hop_length=hop_length)
    logS = librosa.power_to_db(S_mel)
    mfcc = librosa.feature.mfcc(S=logS, sr=sr, n_mfcc=N_MFCC)
    # CMVN por utterance
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    # --- Espectrales clásicos ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).squeeze()
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length).squeeze()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).squeeze()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).squeeze()
    roll85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop_length).squeeze()
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length).squeeze()
    contrast = librosa.feature.spectral_contrast(S=None, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=200.0)

    # --- Onset / Flux & Ritmo ---
    # Evitar warning: calcular onset_strength sobre un S ya calculado con tu n_fft/hop
    onset_env = librosa.onset.onset_strength(S=logS, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames")

    # --- Pitch (F0) con YIN ---
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        f0 = np.zeros(1, dtype=float)

    # --- Chroma / Tonnetz ---
    # chroma_stft: usa nuestro n_fft (ok).
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    except Exception:
        chroma = np.zeros((12, 1), dtype=float)

    # chroma_cqt / tonnetz: si el clip es corto, saltar (evita n_fft internos gigantes y warnings)
    if len(y) >= 8192:
        try:
            y_harm, _ = librosa.effects.hpss(y)
            chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
            tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt, sr=sr)
        except Exception:
            chroma_cqt = np.zeros((12, 1), dtype=float)
            tonnetz = np.zeros((6, 1), dtype=float)
    else:
        chroma_cqt = np.zeros((12, 1), dtype=float)
        tonnetz = np.zeros((6, 1), dtype=float)

    # --- LPC por frames (media + std por coeficiente) ---
    y_pre = librosa.effects.preemphasis(y)
    if len(y_pre) < frame_length:
        y_pre = np.pad(y_pre, (0, frame_length - len(y_pre)))
    frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
    win = np.hanning(frame_length).astype(np.float32)
    lpc_list = []
    for i in range(frames.shape[1]):
        fr = frames[:, i] * win
        if np.max(np.abs(fr)) < 1e-3:
            continue
        try:
            a = librosa.lpc(fr, order=LPC_ORDER)  # (LPC_ORDER+1,)
            lpc_list.append(a.astype(float))
        except np.linalg.LinAlgError:
            pass
    if len(lpc_list) == 0:
        lpc_mat = np.zeros((1, LPC_ORDER + 1), dtype=float)
    else:
        lpc_mat = np.vstack(lpc_list)
    lpc_mean = np.mean(lpc_mat, axis=0)
    lpc_std = np.std(lpc_mat, axis=0)

    # --------- Pooling / estadísticos ----------
    def push_stats_mat(mat, prefix):
        mu = np.mean(mat, axis=1); sd = np.std(mat, axis=1)
        for k in range(len(mu)):
            feature_vector.append(float(mu[k])); feature_names.append(f"{prefix}_mean_{k+1}")
        for k in range(len(sd)):
            feature_vector.append(float(sd[k])); feature_names.append(f"{prefix}_std_{k+1}")

    def push_stats_1d(arr, prefix):
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size == 0: arr = np.array([0.0])
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        qs = np.percentile(arr, [10, 90]) if arr.size > 1 else [float(arr[0]), float(arr[0])]
        med = np.median(arr)
        feature_vector.extend([float(np.mean(arr)), float(np.std(arr)), float(qs[0]), float(qs[1]), float(med)])
        feature_names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_p10", f"{prefix}_p90", f"{prefix}_median"])

    # MFCC (+Δ,+ΔΔ)
    push_stats_mat(mfcc, "mfcc")
    push_stats_mat(d1, "mfcc_d")
    push_stats_mat(d2, "mfcc_dd")

    # Mel (log) por banda
    mel_mu = np.mean(logS, axis=1); mel_sd = np.std(logS, axis=1)
    for k in range(len(mel_mu)):
        feature_vector.append(float(mel_mu[k])); feature_names.append(f"mel_log_mean_{k+1}")
    for k in range(len(mel_sd)):
        feature_vector.append(float(mel_sd[k])); feature_names.append(f"mel_log_std_{k+1}")

    # Spectral contrast
    contrast_mu = np.mean(contrast, axis=1); contrast_sd = np.std(contrast, axis=1)
    for k in range(len(contrast_mu)):
        feature_vector.append(float(contrast_mu[k])); feature_names.append(f"contrast_mean_{k+1}")
    for k in range(len(contrast_sd)):
        feature_vector.append(float(contrast_sd[k])); feature_names.append(f"contrast_std_{k+1}")

    # Chroma STFT
    chroma_mu = np.mean(chroma, axis=1); chroma_sd = np.std(chroma, axis=1)
    for k in range(len(chroma_mu)):
        feature_vector.append(float(chroma_mu[k])); feature_names.append(f"chroma_mean_{k+1}")
    for k in range(len(chroma_sd)):
        feature_vector.append(float(chroma_sd[k])); feature_names.append(f"chroma_std_{k+1}")

    # Tonnetz
    ton_mu = np.mean(tonnetz, axis=1); ton_sd = np.std(tonnetz, axis=1)
    for k in range(len(ton_mu)):
        feature_vector.append(float(ton_mu[k])); feature_names.append(f"tonnetz_mean_{k+1}")
    for k in range(len(ton_sd)):
        feature_vector.append(float(ton_sd[k])); feature_names.append(f"tonnetz_std_{k+1}")

    # Series 1D
    push_stats_1d(rms, "rms")
    push_stats_1d(zcr, "zcr")
    push_stats_1d(centroid, "centroid")
    push_stats_1d(bw, "bandwidth")
    push_stats_1d(roll85, "rolloff85")
    push_stats_1d(flatness, "flatness")
    push_stats_1d(onset_env, "onset_strength")
    push_stats_1d(f0, "f0_yin")

    # Ritmo/tempo + #beats
    feature_vector.append(float(tempo)); feature_names.append("tempo_bpm")
    feature_vector.append(float(len(beats))); feature_names.append("n_beats")

    # LPC mean/std (coef 0..LPC_ORDER)
    for k in range(len(lpc_mean)):
        feature_vector.append(float(lpc_mean[k])); feature_names.append(f"lpc_mean_{k}")
    for k in range(len(lpc_std)):
        feature_vector.append(float(lpc_std[k])); feature_names.append(f"lpc_std_{k}")

    # Duración y #frames (con parámetros dinámicos)
    feature_vector.append(duration_s); feature_names.append("duration_s")
    n_frames = int(1 + max(0, len(y) - frame_length) // hop_length) if len(y) >= frame_length else 1
    feature_vector.append(float(max(n_frames, 1))); feature_names.append("n_frames")

    # --------- Guardar JSON ----------
    out = {
        "filename": path.name,
        "label": label,
        "speaker": speaker,
        "sample_rate": int(sr),
        "frame_length": int(frame_length),
        "hop_length": int(hop_length),
        "n_fft": int(n_fft),
        "feature_vector": [float(v) for v in feature_vector],
        "feature_names": feature_names
    }
    out_path = OUTPUT_DIR / f"{path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n✅ Listo. JSONs en: {OUTPUT_DIR}")
