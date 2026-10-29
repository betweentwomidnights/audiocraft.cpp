#!/usr/bin/env python3
"""Prepare short MelodyFlow posterior/T5 caches using the native inference tools.

Only numpy and gguf are needed; Python never runs a model. Caption text is data.
Source tracks are split before chunking to avoid validation leakage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import wave

import gguf
import numpy as np
from train_cache_windows import window_starts


def read_pcm(path, start, samples):
    with wave.open(str(path), "rb") as wav:
        width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
        wav.setpos(start)
        raw = wav.readframes(samples)
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        audio = ((v ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"unsupported PCM width: {width}")
    return audio.reshape(-1, channels), rate


def tensor(reader, name):
    return next(t.data.astype(np.float32) for t in reader.tensors if t.name == name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--dit", type=Path, required=True)
    p.add_argument("--codec", type=Path, required=True)
    p.add_argument("--t5", type=Path, required=True)
    p.add_argument('--cache-seconds',type=float,default=0,help='encode longer windows (up to 30s) for fresh trainer crops; 0 follows --seconds')
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--windows-per-track", type=int, default=4)
    p.add_argument("--max-tracks", type=int, default=0, help="0 uses all tracks")
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cover-track",action="store_true",help="tile the whole source with cache windows")
    args = p.parse_args()
    if not 0 < args.seconds <= 30 or args.windows_per_track <= 0 or args.max_tracks < 0:
        p.error("invalid duration/windows/max-tracks")
    if not 0 <= args.validation_fraction < 1:
        p.error("validation fraction must be in [0,1)")
    args.seconds = args.cache_seconds or args.seconds
    if not 0 < args.seconds <= 30: p.error("cache seconds must be in (0,30]")
    frames = round(args.seconds * 25)
    if abs(frames / 25 - args.seconds) > 1e-6:
        p.error("seconds must be a multiple of 1/25")
    tracks = sorted(args.dataset.glob("*.wav"))
    if not tracks:
        p.error("no WAV tracks")
    for track in tracks:
        if not track.with_suffix(".txt").is_file():
            p.error(f"missing caption for {track}")
    rng = random.Random(args.seed)
    rng.shuffle(tracks)
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]
    n_val = round(len(tracks) * args.validation_fraction)
    if args.validation_fraction and len(tracks) > 1:
        n_val = min(len(tracks) - 1, max(1, n_val))
    validation = set(tracks[:n_val])
    # Read just the DiT's latent statistics through a memory map.
    reader = gguf.GGUFReader(str(args.dit))
    mean = tensor(reader, "dit.latent_mean").reshape(-1, 1)
    std = tensor(reader, "dit.latent_std").reshape(-1, 1) + 1e-5
    latent_dim = len(mean)
    if not np.isfinite(std).all() or (std <= 0).any():
        p.error("invalid DiT latent statistics")
    # Fingerprint the exact normalization constants, so a different base is detectable.
    stats_id = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        p.error("output directory must be empty")
    for split in ("train", "validation"):
        (args.out / split).mkdir()
    suffix = ".exe" if (args.bin / "ac-vae.exe").exists() else ""
    manifest = {"version": 1, "frames": frames, "seed": args.seed,
                "stats_sha256": stats_id, "dit": str(args.dit.resolve()),
                "codec": str(args.codec.resolve()), "t5": str(args.t5.resolve()), "windows": []}
    with tempfile.TemporaryDirectory(prefix="ac-mf-lora-") as temp:
        tmp = Path(temp)
        for track in tracks:
            prompt = track.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
            split = "validation" if track in validation else "train"
            with wave.open(str(track), "rb") as wav:
                rate, length = wav.getframerate(), wav.getnframes()
            window = round(args.seconds * rate)
            if length < window:
                raise ValueError(f"track is shorter than a training window: {track}")
            (tmp / "caption.txt").write_text(prompt, encoding="utf-8")
            subprocess.run([str(args.bin / ("ac-textenc" + suffix)), "--t5", str(args.t5),
                            "--prompt-file", str(tmp / "caption.txt"), "--out", str(tmp / "hidden.f32"),
                            "--device", args.device], check=True)
            hidden = np.fromfile(tmp / "hidden.f32", dtype="<f4")
            if len(hidden) % 768:
                raise ValueError("T5 cache is not [768, tokens]")
            hidden = hidden.reshape(-1, 768)
            starts = window_starts(length,window,args.windows_per_track,True) if args.cover_track else None
            for i in range(len(starts) if starts is not None else args.windows_per_track):
                # One random crop from each equal segment, covering more than the intro.
                lo = (length - window) * i // args.windows_per_track
                hi = (length - window) * (i + 1) // args.windows_per_track
                start = starts[i] if starts is not None else rng.randint(lo, hi)
                audio, rate = read_pcm(track, start, window)
                # Native WAV reader accepts IEEE float; preserve 24-bit source precision.
                # wave only writes PCM, so emit a minimal float32 RIFF directly.
                import struct
                data = audio.astype("<f4").tobytes()
                channels = audio.shape[1]
                fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * channels * 4, channels * 4, 32)
                (tmp / "window.wav").write_bytes(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
                    struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)
                subprocess.run([str(args.bin / ("ac-vae" + suffix)), "--codec", str(args.codec),
                                "--encode", str(tmp / "window.wav"), "--seconds", str(args.seconds),
                                "--out-latent", str(tmp / "posterior.f32"), "--device", args.device], check=True)
                encoded = np.fromfile(tmp / "posterior.f32", dtype="<f4")
                if len(encoded) != 2 * latent_dim * frames:
                    raise ValueError("codec output shape does not match requested window")
                encoded = encoded.reshape(2 * latent_dim, frames)
                mu, scale = encoded[:latent_dim], encoded[latent_dim:]
                norm_mean = (mu - mean) / std
                norm_std = (np.logaddexp(0, scale) + 1e-4) / std
                name = hashlib.sha256(track.name.encode("utf-8")).hexdigest()[:16] + f"-{i:03d}.gguf"
                dest = args.out / split / name
                writer = gguf.GGUFWriter(str(dest), "ac-melodyflow-train-cache")
                writer.add_uint32("ac.cache.version", 1)
                writer.add_bool("ac.cache.empty_prompt", not prompt)
                writer.add_string("ac.cache.prompt", prompt)
                writer.add_string("ac.cache.caption_transport", "utf8-file")
                writer.add_string("ac.cache.stats_sha256", stats_id)
                writer.add_string("ac.cache.source_id", hashlib.sha256(track.name.encode("utf-8")).hexdigest())
                for n, values in (("mean", norm_mean), ("stddev", norm_std), ("hidden", hidden),
                                  ("normalization_mean", mean.reshape(-1)),
                                  ("normalization_std", std.reshape(-1))):
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite {n} for {track}")
                    writer.add_tensor(n, np.ascontiguousarray(values, dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                manifest["windows"].append({"cache": f"{split}/{name}", "source": str(track.resolve()),
                                            "caption": prompt, "start_seconds": start / rate})
                print(f"{split}: {track.name} @ {start/rate:.2f}s -> {name}", flush=True)
    captions = list(dict.fromkeys(w["caption"] for w in manifest["windows"] if w["cache"].startswith("train/") and w["caption"]))
    (args.out / "prompts.json").write_text(json.dumps({"version":1,"source":{"lora":"training-cache","files":len(tracks)-n_val,"unique_prompts":len(captions)},"dice":{"instrumental":captions}},indent=2,ensure_ascii=False),encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
