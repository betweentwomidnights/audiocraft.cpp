#!/usr/bin/env python3
"""Dump a MelodyFlow velocity prediction for the ac-dit parity check.

Runs the real `FlowModel.forward` -- the same call `FlowModel.generate` makes at every
solver step -- through audiocraft's own conditioning path, so a mismatch here is a genuine
difference and not a harness artifact.

Writes:
  <out>/mf_dit_z.f32         the latent fed to both sides, raw f32 [T, latent_dim]
  <out>/mf_dit_cond.f32      T5 hidden states before output_proj, raw f32 [768, tokens]
  <out>/mf_dit_velocity.npy  the prediction [latent_dim, T]

The latent is drawn from a fixed seed rather than taken from real audio: the DiT sees
normalized latents at every flow step, and a standard normal is the distribution it was
trained on, so it exercises the whole dynamic range without needing the codec.

Needs torch and audiocraft. Run it with terry's venv:
  PYTHONPATH=.../services/melodyflow .../env/Scripts/python.exe tools/dump_mf_dit_refs.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--prompt", default="Lively accordion music with a European folk feeling")
    ap.add_argument("--frames", type=int, default=250, help="latent frames (750 = 30 s)")
    ap.add_argument("--t", type=float, default=0.37, help="flow step")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--uncond", action="store_true",
                    help="the classifier-free guidance null branch: empty prompt, mask all "
                         "zeros, so every text key is -inf and only the zero key survives")
    ap.add_argument("--model", default="facebook/melodyflow-t24-30secs")
    args = ap.parse_args()

    import torch
    from audiocraft.models.loaders import load_dit_model_melodyflow
    from audiocraft.modules.conditioners import ConditioningAttributes

    lm = load_dit_model_melodyflow(args.model, device="cpu")
    lm.eval()

    # Condition exactly the way FlowModel.generate does, through the model's own provider.
    description = None if args.uncond else args.prompt
    attributes = [ConditioningAttributes(text={"description": description})]
    conditioner = lm.condition_provider.conditioners["description"]
    with torch.no_grad():
        tokenized = lm.condition_provider.tokenize(attributes)
        conditions = lm.condition_provider(tokenized)
        # The DiT consumes the projected states; the C++ side runs its own T5 and does the
        # projection itself, so dump what T5 produced before output_proj as well.
        hidden = conditioner.t5(**tokenized["description"]).last_hidden_state[0]
    projected, mask = conditions["description"]      # [1, tokens, dim], [1, tokens]

    torch.manual_seed(args.seed)
    latent_dim = lm.latent_dim
    z = torch.randn(1, latent_dim, args.frames)
    t = torch.tensor([args.t]).expand(1, 1)

    with torch.no_grad():
        velocity = lm.forward(z, t, projected, torch.log(mask.unsqueeze(1).unsqueeze(1)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ggml [T, latent_dim] has the channel as the slow axis, so torch's [latent_dim, T] is
    # already in the right ravel order.
    z[0].numpy().astype(np.float32).tofile(out_dir / "mf_dit_z.f32")
    # T5 hidden states are ggml [cond_dim, tokens] -> numpy [tokens, cond_dim].
    hidden.numpy().astype(np.float32).tofile(out_dir / "mf_dit_cond.f32")
    np.save(out_dir / "mf_dit_velocity.npy", velocity[0].to(torch.float32).numpy())

    print(f"prompt   {description!r} -> {int(mask.sum())} valid / {mask.shape[1]} token(s)")
    print(f"z        {tuple(z[0].shape)} seed {args.seed} -> mf_dit_z.f32")
    print(f"cond     {tuple(hidden.shape)} (pre-projection) -> mf_dit_cond.f32")
    print(f"velocity {tuple(velocity[0].shape)} at t={args.t} -> mf_dit_velocity.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
