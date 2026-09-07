#!/usr/bin/env python3
"""Dump MelodyFlow VAE references for the ac-vae parity check.

Runs audiocraft's real compression model -- the one `MelodyFlow.get_pretrained` builds,
with weight-norm parametrizations removed exactly as it does -- and writes:

  <out>/input48k.wav      the resampled, stereo, hop-aligned input BOTH sides read, so the
                          comparison measures the codec and not two different resamplers
  <out>/mf_vae_latent.npy encoder output [2*latent_dim, T]  (mean||scale)
  <out>/mf_vae_z.f32      the latent to decode, as raw f32 in ggml order [T, latent_dim]
  <out>/mf_vae_audio.npy  decoder(z) [channels, samples]

`z` is the posterior *mean*, not a sample: `vae_sample` draws `randn_like(mean)`, and
matching torch's RNG stream is a separate problem from matching the codec. Feeding both
implementations the same z keeps encode and decode independently falsifiable.

Needs torch, torchaudio and audiocraft. Run it with the melodyflow service venv:
  .../services/melodyflow/env/Scripts/python.exe tools/dump_mf_vae_refs.py --audio x.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def clear_weight_norm(module):
    """What MelodyFlow.get_pretrained does before moving the codec to the device."""
    import torch

    def remove(m):
        if hasattr(m, "conv") and hasattr(m.conv, "conv"):
            torch.nn.utils.parametrize.remove_parametrizations(m.conv.conv, "weight")
        if hasattr(m, "convtr") and hasattr(m.convtr, "convtr"):
            torch.nn.utils.parametrize.remove_parametrizations(m.convtr.convtr, "weight")

    remove(module)
    for child in module.children():
        clear_weight_norm(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="input wav (any rate/channel count)")
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="crop length; terry uses the full 30 s window")
    ap.add_argument("--model", default="facebook/melodyflow-t24-30secs")
    ap.add_argument("--taps", action="store_true",
                    help="also dump each top-level encoder/decoder layer output, for "
                         "bisecting a divergence")
    args = ap.parse_args()

    import torch
    import torchaudio
    from audiocraft.models.loaders import load_compression_model

    model = load_compression_model(args.model, device="cpu")
    model.to("cpu")
    clear_weight_norm(model)
    model.eval()

    sample_rate = model.sample_rate
    channels = model.channels
    frame_rate = model.frame_rate
    hop = round(sample_rate / frame_rate)

    wav, rate = torchaudio.load(args.audio)
    if rate != sample_rate:
        wav = torchaudio.functional.resample(wav, rate, sample_rate)
    if wav.shape[0] == 1:
        wav = wav.repeat(channels, 1)
    elif wav.shape[0] > channels:
        wav = wav[:channels]

    frames = int(args.seconds * frame_rate)
    keep = frames * hop
    if wav.shape[-1] < keep:
        raise SystemExit(f"input is only {wav.shape[-1] / sample_rate:.2f}s, need {args.seconds}s")
    wav = wav[..., :keep].contiguous()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "input48k.wav"), wav, sample_rate,
                    encoding="PCM_S", bits_per_sample=16)
    # Read the saved 16-bit file back, so the reference sees the exact same quantized
    # samples the C++ side will read. Otherwise the comparison starts from different inputs.
    wav, _ = torchaudio.load(str(out_dir / "input48k.wav"))
    x = wav.unsqueeze(0)

    taps = {}
    with torch.no_grad():
        if args.taps:
            h = x
            for i, layer in enumerate(model.encoder.model):
                h = layer(h)
                taps[f"mf_vae_enc{i}"] = h[0].clone()
            latent = h
        else:
            latent = model.encoder(x)

        mean, scale = latent.chunk(2, dim=1)
        z = mean

        if args.taps:
            h = z
            for i, layer in enumerate(model.decoder.model):
                h = layer(h)
                taps[f"mf_vae_dec{i}"] = h[0].clone()
            audio = h
        else:
            audio = model.decoder(z)

    np.save(out_dir / "mf_vae_latent.npy", latent[0].to(torch.float32).numpy())
    np.save(out_dir / "mf_vae_audio.npy", audio[0].to(torch.float32).numpy())
    # ggml order for a [frames, latent_dim] tensor is channel-slowest, so the torch
    # [latent_dim, frames] array is already in the right ravel order.
    z[0].to(torch.float32).numpy().astype(np.float32).tofile(out_dir / "mf_vae_z.f32")

    print(f"input   {tuple(wav.shape)} @ {sample_rate} Hz -> {out_dir / 'input48k.wav'}")
    print(f"latent  {tuple(latent[0].shape)} (mean||scale) -> mf_vae_latent.npy")
    print(f"z       {tuple(z[0].shape)} -> mf_vae_z.f32")
    print(f"audio   {tuple(audio[0].shape)} -> mf_vae_audio.npy")

    if args.taps:
        for name, tensor in taps.items():
            np.save(out_dir / f"{name}.npy", tensor.to(torch.float32).numpy())
        print(f"wrote {len(taps)} layer taps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
