#!/usr/bin/env python3
"""Dump MusicGen's EnCodec 32 kHz in both directions, for the ac-encodec parity check.

Runs the codec audiocraft actually uses. That is not audiocraft's own SEANet: MusicGen's
`compression_state_dict.bin` is a pointer to `facebook/encodec_32khz`, and
`CompressionModel.get_pretrained` loads it through HuggingFace `transformers` as an
`EncodecModel`. Same architecture, different code and different tensor names -- see
docs/MUSICGEN_ENCODEC.md.

Both directions here are deterministic (the RVQ is a nearest-neighbour search, not a
sample), so each can be compared on its own, which is what tells a codec bug apart from an
LM bug when the end-to-end audio is wrong.

Writes into <out>/:
  mg_codec_input.wav      the conformed mono 32 kHz input the model saw
  mg_codec_latent.npy     the encoder output before quantization, [dim, frames]
  mg_codec_codes.i32      the RVQ codes, [n_q, frames]
  mg_codec_quantized.npy  the latent the codes decode back to, [dim, frames]
  mg_codec_audio.npy      the decoded audio, [channels, samples]
  mg_codec_meta.txt       the settings

Needs torch and audiocraft. Run it with gary's venv:
  PYTHONPATH=.../services/gary .../env/Scripts/python.exe tools/dump_mg_codec_refs.py \
      --input testdata/music.wav --out refcodec
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refcodec")
    ap.add_argument("--input", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--from-end", action="store_true",
                    help="take the tail rather than the head, as gary's continue_music does")
    ap.add_argument("--model", default="facebook/encodec_32khz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torchaudio
    from audiocraft.models.encodec import CompressionModel

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CompressionModel.get_pretrained(args.model, device=args.device)
    model.eval()
    sample_rate, channels = model.sample_rate, model.channels
    frame_rate = model.frame_rate
    hop = round(sample_rate / frame_rate)

    waveform, sr = torchaudio.load(args.input)
    # Order matters and follows gary: `resample_for_model` runs on whatever channel layout
    # arrived, and `safe_musicgen_continuation_v2` mixes to mono afterwards. Resampling a
    # mono mix is not the same computation as mixing two resampled channels.
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] != channels:
        waveform = waveform.mean(dim=0, keepdim=True).repeat(channels, 1)
    take = int(args.seconds * sample_rate)
    take -= take % hop
    if waveform.shape[-1] > take:
        waveform = waveform[..., -take:] if args.from_end else waveform[..., :take]
    frames = waveform.shape[-1] // hop
    torchaudio.save(str(out_dir / "mg_codec_input.wav"), waveform, sample_rate,
                    encoding="PCM_F", bits_per_sample=32)

    device = args.device
    x = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        # The encoder output before quantization. `HFEncodecCompressionModel` does not
        # expose it, so reach through to the wrapped model -- it is the checkpoint that
        # tells a conv-stack divergence apart from a codebook one.
        latent = model.model.encoder(x)
        codes, scale = model.encode(x)
        if scale is not None:
            raise SystemExit("this checkpoint normalizes; the C++ side does not carry a scale")
        quantized = model.decode_latent(codes)
        audio = model.decode(codes)

    np.save(out_dir / "mg_codec_latent.npy", latent[0].float().cpu().numpy())
    codes[0].to(torch.int32).cpu().numpy().tofile(out_dir / "mg_codec_codes.i32")
    np.save(out_dir / "mg_codec_quantized.npy", quantized[0].float().cpu().numpy())
    np.save(out_dir / "mg_codec_audio.npy", audio[0].float().cpu().numpy())

    n_q = int(codes.shape[1])
    (out_dir / "mg_codec_meta.txt").write_text(
        f"model={args.model}\ndevice={args.device}\nsample_rate={sample_rate}\n"
        f"channels={channels}\nframe_rate={frame_rate}\nhop={hop}\nframes={frames}\n"
        f"n_q={n_q}\ncardinality={model.cardinality}\nseconds={args.seconds}\n"
        f"from_end={args.from_end}\n")

    print(f"model    {args.model} on {args.device}: {sample_rate} Hz, {channels}ch, "
          f"{frame_rate} Hz frames, {n_q} codebooks of {model.cardinality}")
    print(f"input    {tuple(waveform.shape)} -> {frames} frames")
    print(f"latent   {tuple(latent[0].shape)} -> mg_codec_latent.npy")
    print(f"codes    {tuple(codes[0].shape)} -> mg_codec_codes.i32")
    print(f"audio    {tuple(audio[0].shape)} -> mg_codec_audio.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
