"""Regenerate the fixtures embedded in tests/codec_test.cpp.

Run with gary's venv so torchaudio and transformers are importable:
  PYTHONPATH=.../services/gary .../env/Scripts/python.exe tests/gen_codec_fixture.py

Two things get pinned here, both of which were wrong at some point during Phase 5:

  * the residual vector quantizer, against `transformers`' own `EncodecResidualVectorQuantizer`
  * the resampler, against `torchaudio.functional.resample` -- including the odd fact that
    torchaudio's output length goes through a float32 round trip and can therefore be one
    sample shorter than the true ceiling
"""
import math
import sys

import numpy as np
import torch


def carr(name, values, kind="float"):
    lines, cur = [], "        "
    vals = list(values)
    for i, v in enumerate(vals):
        if kind == "float":
            text = f"{float(v):.9g}"
            if "." not in text and "e" not in text and "n" not in text:
                text += ".0"
            text += "f"
        else:
            text = f"{int(v)}"
        piece = text + ("," if i + 1 < len(vals) else "")
        if len(cur) + len(piece) > 92:
            lines.append(cur.rstrip()); cur = "        "
        cur += piece + " "
    lines.append(cur.rstrip())
    ctype = "float" if kind == "float" else "int32_t"
    return f"    static const {ctype} {name}[] = {{\n" + "\n".join(lines) + "\n    };"


def emit_rvq():
    """The real `EncodecResidualVectorQuantizer` loop over a miniature codebook set."""
    from transformers.models.encodec.modeling_encodec import (
        EncodecConfig, EncodecResidualVectorQuantizer)

    n_q, bins, dim, frames = 3, 16, 8, 7
    # `num_quantizers` is derived, not set, and the two derivations differ:
    # `EncodecConfig.num_quantizers` hardcodes ten bits per codebook
    # (`1000 * bw // (frame_rate * 10)`) while audiocraft's own wrapper uses
    # `log2(cardinality)`. For the shipped 32 kHz model both give 4 and the difference never
    # surfaces; here it decides how many layers get built, so 1.5 kbps at 50 Hz asks HF for
    # three. tools/convert_seanet.py follows audiocraft's, because that is what selects the
    # codebook count at inference.
    config = EncodecConfig(codebook_size=bins, codebook_dim=dim, hidden_size=dim,
                           target_bandwidths=[1.5], sampling_rate=32000,
                           upsampling_ratios=[8, 5, 4, 4])
    quantizer = EncodecResidualVectorQuantizer(config)
    g = torch.Generator().manual_seed(23)
    for i, layer in enumerate(quantizer.layers):
        layer.codebook.embed.copy_(torch.randn(bins, dim, generator=g))
    latent = torch.randn(1, dim, frames, generator=g) * 1.5

    with torch.no_grad():
        codes = quantizer.encode(latent)          # [n_q, B, frames]
        quantized = quantizer.decode(codes)       # [B, dim, frames]

    print(f"    // RVQ: {n_q} codebooks of {bins} x {dim}, {frames} frames")
    print(f"    constexpr int RVQ_NQ = {n_q}, RVQ_BINS = {bins}, RVQ_DIM = {dim}, "
          f"RVQ_FRAMES = {frames};")
    # ggml stores each codebook as [dim, bins], which is row-major [bins][dim].
    embeds = torch.cat([l.codebook.embed for l in quantizer.layers]).numpy().ravel()
    print(carr("rvq_embed", embeds))
    # Latents are ggml [frames, dim] -- channel-major, which is torch's [dim, frames] ravel.
    print(carr("rvq_latent", latent[0].numpy().ravel()))
    print(carr("rvq_codes", codes[:, 0].numpy().ravel(), kind="int"))
    print(carr("rvq_quantized", quantized[0].numpy().ravel()))


def emit_resample():
    import torchaudio

    print("    // Resampling, against torchaudio.functional.resample (sinc_interp_hann).")
    cases = [("R44", 44100, 32000, 2000), ("R48", 48000, 32000, 1500),
             ("R32", 32000, 44100, 1000)]
    for tag, src, dst, n in cases:
        rng = np.random.default_rng(hash(tag) % 2**31)
        x = (rng.standard_normal(n) * 0.3).astype(np.float32)
        y = torchaudio.functional.resample(torch.from_numpy(x), src, dst).numpy()
        print(f"    // {tag}: {src} -> {dst}, {n} samples -> {len(y)}")
        print(f"    constexpr int {tag}_SRC = {src}, {tag}_DST = {dst}, "
              f"{tag}_N = {n}, {tag}_OUT = {len(y)};")
        print(carr(f"{tag}_in", x))
        print(carr(f"{tag}_out", y))

    # The length trap, stated as data: torchaudio's target length goes through float32, so
    # for these it is one short of ceil(new * n / orig).
    print("    // torchaudio's output length, against the exact ceiling.")
    lengths = []
    for n in (5443201, 1000003, 999983):
        got = int(torch.ceil(torch.as_tensor(320 * n / 441)).long())
        exact = math.ceil(320 * n / 441)
        lengths.append((n, got, exact))
    print(f"    constexpr int LENGTH_CASES = {len(lengths)};")
    print(carr("length_n", [n for n, _, _ in lengths], kind="int"))
    print(carr("length_torchaudio", [g for _, g, _ in lengths], kind="int"))
    print(carr("length_exact_ceil", [e for _, _, e in lengths], kind="int"))


def main():
    emit_rvq()
    emit_resample()
    return 0


if __name__ == "__main__":
    sys.exit(main())
