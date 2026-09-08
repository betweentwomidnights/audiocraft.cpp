#!/usr/bin/env python3
"""Dump a MusicGen generation for the mg-generate parity check.

Runs audiocraft's own `LMModel.generate` -- the delay pattern, the KV-cached streaming
transformer, the guidance combination -- and writes the codes it produces alongside the
prompt codes it started from, so the C++ side can be compared token for token.

**Use greedy decoding for that comparison.** `--sampled` exists for listening checks, but
sampling depends on the RNG stream and ours is not torch's, so a seed does not carry across.
Greedy is the only mode in which two correct implementations must agree exactly.

Device matters. `load_lm_model` builds the model in float32 on CPU and **float16** on CUDA,
and `MusicGen` wraps generation in a float16 autocast on CUDA -- so torch on GPU, which is
what gary runs, is a different computation from torch on CPU. Dump on CPU to compare
arithmetic; dump on CUDA to see what the service actually emits.

Writes into <out>/:
  mg_prompt_codes.i32     the EnCodec codes the continuation started from, [n_q, prompt_len]
  mg_codes.i32            the generated codes, [n_q, timesteps], prompt included
  mg_prompt.wav           the conformed audio prompt, when there was one
  mg_logits0.npy          the first prediction's combined logits, [n_q, card]
  mg_meta.txt             the settings, so a mismatched comparison is obvious

Needs torch and audiocraft. Run it with gary's venv:
  PYTHONPATH=.../services/gary .../env/Scripts/python.exe tools/dump_mg_lm_refs.py --out refmg
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refmg")
    ap.add_argument("--model", default="thepatch/vanya_ai_dnb_0.1")
    ap.add_argument("--prompt", default="", help="text conditioning")
    ap.add_argument("--input", help="audio to continue from; omit for text-to-music")
    ap.add_argument("--prompt-duration", type=float, default=6.0,
                    help="seconds of --input to use as the prompt (gary's default)")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--sampled", action="store_true",
                    help="top-k sampling instead of greedy; not comparable token for token")
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg-coef", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torchaudio
    from audiocraft.models.musicgen import MusicGen

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MusicGen.get_pretrained(args.model, device=args.device)
    model.set_generation_params(
        use_sampling=args.sampled,
        top_k=args.top_k,
        top_p=0.0,
        temperature=args.temperature,
        duration=args.duration,
        cfg_coef=args.cfg_coef,
    )

    # Capture what `_generate_tokens` was handed: the prompt codes are produced inside
    # `_prepare_tokens_and_attributes` by the compression model and are not otherwise
    # reachable, and they are exactly what mg-generate needs as input.
    seen = {}
    real_generate_tokens = model._generate_tokens

    def traced(attributes, prompt_tokens, progress=False):
        seen["prompt_tokens"] = prompt_tokens
        return real_generate_tokens(attributes, prompt_tokens, progress)

    model._generate_tokens = traced

    # The first prediction's combined logits. `_sample_next_token` runs the two guidance
    # branches as one batch-2 forward and combines them as
    # `uncond + (cond - uncond) * cfg_coef`; capturing the forward and repeating that here
    # gives exactly the vector the C++ `--out-logits` writes. It is the smallest thing the
    # two implementations can disagree about, which is what makes it worth dumping: greedy
    # decoding turns any disagreement into a different song within a few steps.
    real_forward = model.lm.forward

    def traced_forward(sequence, *rest, **kwargs):
        out = real_forward(sequence, *rest, **kwargs)
        if "logits0" not in seen:
            last = out[..., -1, :]                       # [B, K, card]
            if last.shape[0] == 2:
                cond_l, uncond_l = last[0], last[1]
                seen["logits0"] = uncond_l + (cond_l - uncond_l) * args.cfg_coef
            else:
                seen["logits0"] = last[0]
        return out

    model.lm.forward = traced_forward
    torch.manual_seed(args.seed)
    import time
    started = time.perf_counter()
    try:
        if args.input:
            waveform, sr = torchaudio.load(args.input)
            # gary reduces to mono and takes the tail of the track as the prompt.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != model.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, model.sample_rate)
            take = int(args.prompt_duration * model.sample_rate)
            waveform = waveform[..., -take:] if waveform.shape[-1] > take else waveform
            torchaudio.save(str(out_dir / "mg_prompt.wav"), waveform, model.sample_rate,
                            encoding="PCM_F", bits_per_sample=32)
            _, codes = model.generate_continuation(
                waveform.unsqueeze(0).to(next(model.lm.parameters()).device),
                prompt_sample_rate=model.sample_rate,
                descriptions=[args.prompt] if args.prompt else None,
                progress=True, return_tokens=True)
        else:
            _, codes = model.generate([args.prompt], progress=True, return_tokens=True)
    finally:
        elapsed = time.perf_counter() - started
        model._generate_tokens = real_generate_tokens
        model.lm.forward = real_forward

    prompt_tokens = seen.get("prompt_tokens")
    prompt_len = 0 if prompt_tokens is None else int(prompt_tokens.shape[-1])
    if prompt_tokens is not None:
        prompt_tokens[0].to(torch.int32).cpu().numpy().tofile(out_dir / "mg_prompt_codes.i32")
    codes[0].to(torch.int32).cpu().numpy().tofile(out_dir / "mg_codes.i32")
    np.save(out_dir / "mg_logits0.npy", seen["logits0"].float().cpu().numpy())

    n_q, timesteps = int(codes.shape[1]), int(codes.shape[2])
    meta = (f"model={args.model}\nprompt={args.prompt!r}\ndevice={args.device}\n"
            f"duration={args.duration}\nframe_rate={model.frame_rate}\n"
            f"decoding={'sampled' if args.sampled else 'greedy'}\n"
            f"top_k={args.top_k}\ntemperature={args.temperature}\ncfg_coef={args.cfg_coef}\n"
            f"seed={args.seed}\nn_q={n_q}\ntimesteps={timesteps}\nprompt_len={prompt_len}\n")
    (out_dir / "mg_meta.txt").write_text(meta)

    print(f"model       {args.model} on {args.device}")
    print(f"prompt      {prompt_len} timestep(s) -> mg_prompt_codes.i32")
    print(f"codes       [{n_q}, {timesteps}] at {model.frame_rate} Hz -> mg_codes.i32")
    print(f"decoding    {'sampled' if args.sampled else 'greedy'}, cfg {args.cfg_coef}")
    print(f"generate    {elapsed:.2f}s on {args.device} (the LM loop plus the codec decode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
