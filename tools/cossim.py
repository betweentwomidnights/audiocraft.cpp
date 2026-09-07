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
    "mf_vae_latent":  (1, 0),   # torch [256, T]  (mean||scale)
    "mf_vae_audio":   (0, 1),   # torch [2, samples]
    "mf_dit_velocity": (1, 0),  # torch [128, T]
    "mf_edit_latent": (1, 0),   # torch [128, T]

    # --- MusicGen (phases 4-5) ---
    "mg_lm_logits":   (1, 0),   # torch [K*card, T]
    "mg_encodec_emb": (1, 0),   # torch [128, T]
    "mg_audio":       (0, 1),   # torch [1, samples]
}


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
