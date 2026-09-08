#!/usr/bin/env python3
"""Compare C++ raw-f32 dumps against the PyTorch .npy references.

The C++ tools write each checkpoint in GGML memory order (ne0 fastest). For each
checkpoint we transpose the PyTorch reference into that same order, then report cosine
similarity and max abs error.

Adapted from sa3.cpp tools/cossim.py; the LAYOUT table is audiocraft's.

Usage:
  python tools/cossim.py --ref refdata --cpp cppout [--gate 0.9999]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# checkpoint -> np.transpose axes that turn the (batch-dropped) PyTorch array into the C++
# ggml ravel order. ggml ne=[a,b,...] memory == numpy shape [..., b, a] in C order.
LAYOUT = {
    # --- shared text conditioner ---
    # torch [seq, 768]; ggml [768, seq] with ne0=dim -> ravel (seq, dim) == torch C order.
    "t5_hidden": (0, 1),

    # --- MelodyFlow (phases 1-3) ---
    # Every one of these is channel-slowest on both sides: a ggml [T, C] tensor is numpy
    # [C, T] in C order, which is already how torch holds them. No transpose.
    "mf_vae_input":    (0, 1),  # torch [channels, samples]
    "mf_vae_enc_pre":  (1, 0),  # ggml transposes to [C, T] for the host LSTM
    "mf_vae_enc_lstm": (1, 0),
    "mf_vae_dec_pre":  (1, 0),
    "mf_vae_dec_lstm": (1, 0),
    "mf_vae_latent":   (0, 1),  # torch [2*latent_dim, T] (mean||scale)
    "mf_vae_audio":    (0, 1),  # torch [channels, samples]
    "mf_dit_velocity": (0, 1),  # torch [latent_dim, T]
    "mf_edit_prompt_latent": (0, 1),  # torch [latent_dim, T], normalized
    "mf_edit_intermediate":  (0, 1),  # the latent at the pivot between the two passes
    "mf_edit_latent":  (0, 1),  # torch [latent_dim, T]
    "mf_edit_velocity0": (0, 1),  # the first velocity of the inversion
    "mf_edit_audio":   (0, 1),  # torch [channels, samples]

    # --- MusicGen (phases 4-5) ---
    "mg_logits0":     (0, 1),   # torch [n_q, card]: the first prediction, guidance applied
    "mg_encodec_emb": (0, 1),   # torch [128, T]
    "mg_audio":       (0, 1),   # torch [channels, samples]
}


def framed(x, win=1024, hop=512):
    """[channels, samples] -> [channels, frames, win], zero-padded to a whole frame."""
    n = x.shape[-1]
    frames = max(1, 1 + (n - win) // hop) if n >= win else 1
    out = np.zeros((x.shape[0], frames, win))
    for i in range(frames):
        chunk = x[:, i * hop:i * hop + win]
        out[:, i, :chunk.shape[-1]] = chunk
    return out


def perceptual(ref, cpp, channels):
    """RMS-envelope and log-magnitude-spectrum cosines.

    Raw waveform cosine is the strict measure and the one that gates, but it punishes a
    sample of phase drift as hard as a missing instrument. For judging precision tiers --
    F16 against F32, and later the quantized ones -- these two say whether the difference
    is audible, which is the question actually being asked. Same convention as sa3.cpp's
    docs/STABLE_AUDIO_OPEN_SMALL.md.
    """
    a = ref.reshape(channels, -1)
    b = cpp.reshape(channels, -1)
    fa, fb = framed(a), framed(b)
    rms_a = np.sqrt((fa ** 2).mean(axis=-1)).ravel()
    rms_b = np.sqrt((fb ** 2).mean(axis=-1)).ravel()
    spec_a = np.log1p(np.abs(np.fft.rfft(fa * np.hanning(fa.shape[-1]), axis=-1))).ravel()
    spec_b = np.log1p(np.abs(np.fft.rfft(fb * np.hanning(fb.shape[-1]), axis=-1))).ravel()
    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-20))
    return cos(rms_a, rms_b), cos(spec_a, spec_b)


def load_ref(p):
    a = np.load(p).astype(np.float64)
    if a.ndim > 1 and a.shape[0] == 1:  # drop batch
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="refdata")
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--gate", type=float, default=0.9999,
                    help="minimum acceptable cosine similarity (default 0.9999)")
    ap.add_argument("--channels", type=int, default=2,
                    help="channel count for the perceptual measures on *_audio dumps")
    args = ap.parse_args()
    ref_dir, cpp_dir = Path(args.ref), Path(args.cpp)

    worst = 1.0
    compared = 0
    for name, axes in LAYOUT.items():
        rp, cp = ref_dir / f"{name}.npy", cpp_dir / f"{name}.f32"
        if not cp.exists() or not rp.exists():
            continue
        ref = np.transpose(load_ref(rp), axes).ravel()
        cpp = np.fromfile(cp, dtype=np.float32).astype(np.float64)
        if ref.size != cpp.size:
            print(f"  !! {name:18s} SIZE MISMATCH ref={ref.size} cpp={cpp.size}")
            worst = 0.0
            compared += 1
            continue
        cos = float(np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-20))
        maxerr = float(np.max(np.abs(ref - cpp)))
        flag = "OK " if cos > args.gate else "!! "
        print(f"  {flag}{name:18s} cossim={cos:.7f}  max_abs_err={maxerr:.3e}  n={ref.size}")
        if name.endswith("_audio"):
            rms, spec = perceptual(ref, cpp, args.channels)
            print(f"     {'':18s} rms-envelope={rms:.7f}  log-spectrum={spec:.7f}")
        worst = min(worst, cos)
        compared += 1

    if not compared:
        print(f"no checkpoints compared (ref={ref_dir}, cpp={cpp_dir})")
        return 1
    print(f"\nworst cossim: {worst:.7f}  ->  {'PASS' if worst > args.gate else 'FAIL'}"
          f"  (gate {args.gate})")
    return 0 if worst > args.gate else 1


if __name__ == "__main__":
    sys.exit(main())
