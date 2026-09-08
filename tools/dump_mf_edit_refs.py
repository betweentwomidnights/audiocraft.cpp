#!/usr/bin/env python3
"""Dump a whole `MelodyFlow.edit` for the mf-edit parity check.

Runs terry's transform through audiocraft unmodified -- the real `edit`, the real
`FlowModel.generate`, the real codec -- and writes every stage the C++ side also produces.

The one thing that is *not* left alone is the noise. `edit` draws from the global torch RNG
twice: once sampling the VAE posterior of the encoded input, then twice per regularized
solver step. Those streams cannot be reproduced in C++, so instead of trying, this script
generates the noise itself, writes it out, and patches `torch.randn_like` to hand it back.
`mf-edit --noise` replays the same file, which leaves float arithmetic as the only
difference between the two runs.

Writes into <out>/:
  mf_edit_input.wav            the conformed input the model actually saw
  mf_edit_noise.f32            the noise stream, in consumption order
  mf_edit_prompt_latent.npy    the normalized latent entering the inversion [latent_dim, T]
  mf_edit_intermediate.npy     the latent at the pivot, between the two passes
  mf_edit_latent.npy           the edited latent, still normalized
  mf_edit_audio.npy            the decoded audio [channels, samples]

Needs torch and audiocraft. Run it with terry's venv:
  PYTHONPATH=.../services/melodyflow .../env/Scripts/python.exe tools/dump_mf_edit_refs.py \
      --input refdata30/input48k.wav --prompt "..." --out refedit
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refedit")
    ap.add_argument("--input", required=True, help="source audio (any rate, any channels)")
    ap.add_argument("--prompt", default="Lively accordion music with a European folk feeling")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="crop the input; 0 means the model's full window")
    ap.add_argument("--solver", default="euler", choices=["euler", "midpoint"])
    ap.add_argument("--steps", type=int, default=0, help="0 picks terry's default per solver")
    ap.add_argument("--flowstep", type=float, default=0.12)
    ap.add_argument("--regularize-iters", type=int, default=2)
    ap.add_argument("--keep-last-k", type=int, default=1)
    ap.add_argument("--lambda-kl", type=float, default=0.2)
    ap.add_argument("--no-regularize", action="store_true",
                    help="run the inversion without the KL term, to separate the solver "
                         "loop from the regularization when bisecting a divergence")
    ap.add_argument("--trace-forwards", type=int, default=0,
                    help="dump the input, flow step and output of the first N DiT forwards")
    ap.add_argument("--seed", type=int, default=1234, help="seeds the dumped noise stream")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="facebook/melodyflow-t24-30secs")
    args = ap.parse_args()

    import torch
    import torchaudio
    from audiocraft.models.melodyflow import MelodyFlow

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MelodyFlow.get_pretrained(args.model, device=args.device)
    regularize = args.solver == "euler" and not args.no_regularize
    steps = args.steps if args.steps > 0 else (25 if regularize else 64)
    model.set_editing_params(
        solver=args.solver,
        steps=steps,
        target_flowstep=args.flowstep,
        regularize=regularize,
        regularize_iters=args.regularize_iters if regularize else 0,
        keep_last_k_iters=args.keep_last_k if regularize else 0,
        lambda_kl=args.lambda_kl if regularize else 0.0,
    )

    # --- the input, conformed exactly the way the service does ---------------------------
    waveform, sr = torchaudio.load(args.input)
    if sr != model.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, model.sample_rate)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    samples_per_latent = round(model.sample_rate / model.frame_rate)
    max_latents = int(model.max_duration * model.frame_rate)
    if args.seconds > 0:
        max_latents = min(max_latents, int(args.seconds * model.frame_rate))
    waveform = waveform[..., :max_latents * samples_per_latent]
    frames = waveform.shape[-1] // samples_per_latent
    device = next(model.lm.parameters()).device
    waveform = waveform.unsqueeze(0).to(device)
    # 32-bit float, so mf-edit reads back exactly what the model saw. The default 16-bit
    # PCM would quantize the input and make every downstream comparison approximate.
    torchaudio.save(str(out_dir / "mf_edit_input.wav"), waveform[0].cpu(), model.sample_rate,
                    encoding="PCM_F", bits_per_sample=32)

    tokens = model.encode_audio(waveform)

    # --- the noise stream -----------------------------------------------------------------
    #
    # One draw for the posterior sample, then two per regularized inner iteration. Generated
    # here rather than by torch so the same values can be replayed on the other side.
    latent_dim = model.lm.latent_dim
    threshold = args.regularize_iters - args.keep_last_k
    kl_iters = max(0, args.regularize_iters - threshold) if regularize else 0
    n_draws = 1 + 2 * kl_iters * steps
    rng = np.random.default_rng(args.seed)
    stream = rng.standard_normal((n_draws, latent_dim, frames), dtype=np.float32)
    stream.tofile(out_dir / "mf_edit_noise.f32")

    drawn = {"at": 0}

    def replay(x, *unused, **kwargs):
        i = drawn["at"]
        if i >= n_draws:
            raise RuntimeError(f"the model wanted more than the {n_draws} predicted draws")
        want = (1, latent_dim, frames)
        if tuple(x.shape) != want:
            raise RuntimeError(f"draw {i} has shape {tuple(x.shape)}, expected {want}")
        drawn["at"] = i + 1
        return torch.from_numpy(stream[i]).unsqueeze(0).to(x.device, x.dtype)

    # --- trace the two passes ---------------------------------------------------------------
    passes = []
    real_generate_tokens = model._generate_tokens

    def traced_generate_tokens(**kwargs):
        result = real_generate_tokens(**kwargs)
        passes.append(result)
        return result

    first_input = {}
    forwards = []
    real_forward = model.lm.forward

    def traced_forward(z, *rest, **kwargs):
        # The very first call sees `gen_sequence` untouched, which is the normalized prompt.
        out = real_forward(z, *rest, **kwargs)
        if "z" not in first_input:
            first_input["z"] = z.detach().float().cpu().numpy()
            first_input["t"] = float(rest[0].reshape(-1)[0]) if rest else 0.0
            first_input["velocity"] = out.detach().float().cpu().numpy()
        if len(forwards) < args.trace_forwards:
            forwards.append((z.detach().float().cpu().numpy(),
                             float(rest[0].reshape(-1)[0]) if rest else 0.0,
                             out.detach().float().cpu().numpy()))
        return out

    saved_randn_like = torch.randn_like
    torch.randn_like = replay
    model._generate_tokens = traced_generate_tokens
    model.lm.forward = traced_forward
    import time
    started = time.perf_counter()
    try:
        audio, final = model.edit(prompt_tokens=tokens, descriptions=[args.prompt],
                                  src_descriptions=[""], progress=True, return_tokens=True)
    finally:
        elapsed = time.perf_counter() - started
        torch.randn_like = saved_randn_like
        model._generate_tokens = real_generate_tokens
        model.lm.forward = real_forward

    if len(passes) != 2:
        raise RuntimeError(f"expected an inversion and a generation, saw {len(passes)} pass(es)")
    if drawn["at"] != n_draws:
        raise RuntimeError(f"the model drew {drawn['at']} times, not the predicted {n_draws}")

    np.save(out_dir / "mf_edit_prompt_latent.npy", first_input["z"][0])
    # The first velocity, with the flow step it was taken at: the smallest thing that can
    # disagree, and the one to check first when a whole solve diverges.
    np.save(out_dir / "mf_edit_velocity0.npy", first_input["velocity"][0])
    (out_dir / "mf_edit_flowstep0.txt").write_text(repr(first_input["t"]))
    # Every traced forward, for walking a divergence call by call: the batch dimension is
    # 1 for the inversion and 2 (conditional then null) once guidance is on.
    lines = []
    for i, (z, t, v) in enumerate(forwards):
        z.astype(np.float32).tofile(out_dir / f"mf_edit_fwd{i}_z.f32")
        np.save(out_dir / f"mf_edit_fwd{i}_v.npy", v)
        lines.append(f"{i} t={t!r} batch={z.shape[0]}")
    if lines:
        (out_dir / "mf_edit_forwards.txt").write_text("\n".join(lines))
    np.save(out_dir / "mf_edit_intermediate.npy", passes[0][0].float().cpu().numpy())
    np.save(out_dir / "mf_edit_latent.npy", final[0].float().cpu().numpy())
    np.save(out_dir / "mf_edit_audio.npy", audio[0].float().cpu().numpy())

    print(f"input       {tuple(waveform.shape[1:])} -> {frames} frames")
    print(f"edit        {elapsed:.2f}s on {device} (both solves plus the decode)")
    print(f"noise       {n_draws} draw(s) x {latent_dim} x {frames} -> mf_edit_noise.f32")
    print(f"settings    {args.solver}/{steps}/{args.flowstep}/"
          f"{args.regularize_iters}/{args.keep_last_k}/{args.lambda_kl}")
    print(f"latents     {tuple(final[0].shape)} -> mf_edit_latent.npy")
    print(f"audio       {tuple(audio[0].shape)} -> mf_edit_audio.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
