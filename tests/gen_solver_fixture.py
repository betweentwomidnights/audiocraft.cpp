"""Regenerate the fixtures embedded in tests/solver_test.cpp.

Run with terry's venv so `audiocraft` is importable:
  PYTHONPATH=.../services/melodyflow .../env/Scripts/python.exe tests/gen_solver_fixture.py

The solve fixtures are produced by calling the *real* `FlowModel.generate` as an unbound
function against a stub `self`. That runs audiocraft's own loop -- schedule, regularization
threshold, moving average, midpoint bookkeeping, CFG batching -- over a velocity field we
choose, so the fixture pins the control flow rather than a transcription of it. The field
is affine and depends on the cross-attention mask, so the conditional and unconditional
branches differ and guidance is actually exercised.
"""
import math
import sys

import numpy as np
import torch
from audiocraft.models.flow import FlowModel
from audiocraft.utils.renoise import noise_regularization


def schedule(steps, source, target, sway=-0.8):
    """audiocraft/models/flow.py::FlowModel.generate, transcribed."""
    if target > source:
        s = torch.arange(0, 1 + 1e-5, 1 / steps)
    else:
        s = torch.arange(1, 0 - 1e-5, -1 / steps)
    s = s + sway * (torch.cos(math.pi * 0.5 * s) - 1 + s)
    if target > source:
        s = s * (target - source) + source
    else:
        s = s * (source - target) + target
    return s


def carr(name, values):
    lines, cur = [], "        "
    vals = list(values)
    for i, v in enumerate(vals):
        text = f"{v:.9g}"
        if "." not in text and "e" not in text and "n" not in text:
            text += ".0"
        piece = text + "f" + ("," if i + 1 < len(vals) else "")
        if len(cur) + len(piece) > 92:
            lines.append(cur.rstrip()); cur = "        "
        cur += piece + " "
    lines.append(cur.rstrip())
    return f"    static const float {name}[] = {{\n" + "\n".join(lines) + "\n    };"


# --- the stub model -------------------------------------------------------------------------
#
# `FlowModel.generate` touches only a handful of attributes on `self`, all of them listed
# here. Everything it does with them -- the batching, the repeat_interleave, the guidance
# combination -- runs unmodified.

C, T = 8, 8
TOKENS = 3


class StubConditionProvider:
    """Stands in for the T5 conditioner: one token embedding per condition, and a mask that
    is all zeros for the null condition (which is what `mask[empty_idx, :] = 0` produces)."""

    def tokenize(self, conditions):
        return conditions

    def __call__(self, conditions):
        n = len(conditions)
        embeds = torch.zeros(n, TOKENS, 4)
        mask = torch.ones(n, TOKENS)
        for i, cond in enumerate(conditions):
            text = cond.text.get("description")
            if text is None or text == "":
                mask[i, :] = 0
        return {"description": (embeds, mask)}


class StubFlow:
    """The velocity field is `A z + B`, scaled by whether the condition survived. Affine in
    z so a step is exactly predictable, but with enough structure (a full [C, T] A and B)
    that a transposed or mis-strided implementation cannot pass by accident."""

    training = False
    cfg_coef = 4.0
    latent_dim = C

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.latent_mean = torch.zeros(1, C, 1)
        self.latent_std = torch.ones(1, C, 1)
        self.A = torch.randn(C, T, generator=g) * 0.3
        self.B = torch.randn(C, T, generator=g) * 0.1
        self.condition_provider = StubConditionProvider()

    def parameters(self):
        return iter([torch.zeros(1)])

    def forward(self, z, t, condition_src, condition_mask):
        # condition_mask is log(mask) shaped [B, 1, 1, tokens]: -inf where the condition was
        # dropped. A conditional row is scaled by 1, an unconditional one by 0.5.
        alive = torch.isfinite(condition_mask[:, 0, 0, 0]).float()
        gain = (0.5 + 0.5 * alive).view(-1, 1, 1)
        step = t.view(-1, 1, 1)
        return gain * (self.A * z + self.B * (1.0 + step))


def recording_randn_like(draws):
    real = torch.randn_like

    def patched(x, *args, **kwargs):
        out = real(x, *args, **kwargs)
        draws.append(out.reshape(-1).numpy().copy())
        return out

    return patched


def solve(prompt, conditions, **kwargs):
    """Run the real generate() loop, returning (result, recorded noise draws)."""
    model = StubFlow()
    draws = []
    torch.manual_seed(3)
    saved = torch.randn_like
    torch.randn_like = recording_randn_like(draws)
    try:
        out = FlowModel.generate(model, prompt=prompt, conditions=conditions, **kwargs)
    finally:
        torch.randn_like = saved
    return out, draws


def attrs(*texts):
    from audiocraft.modules.conditioners import ConditioningAttributes
    return [ConditioningAttributes(text={"description": t}) for t in texts]


def emit_solve(name, prompt, conditions, **kwargs):
    out, draws = solve(prompt, conditions, **kwargs)
    print(f"    // {name}: {kwargs}")
    if draws:
        print(carr(f"{name}_noise", np.concatenate(draws)))
    print(carr(f"{name}_expected", out[0].detach().numpy().ravel()))
    return len(draws)


def main():
    print("// --- schedules: terry's inversion (1.0 -> 0.12) and generation (0.12 -> 1.0) ---")
    print(carr("inversion_schedule", schedule(25, 1.0, 0.12).tolist()))
    print(carr("generation_schedule", schedule(25, 0.12, 1.0).tolist()))
    print("// --- a uniform schedule, sway 0, to pin the mapping without the warp ---")
    print(carr("uniform_schedule", schedule(4, 0.0, 1.0, sway=0.0).tolist()))

    # noise_regularization over a small but structurally faithful case: channels a multiple
    # of 4, frames deliberately NOT a multiple of 4 so the dropped tail is covered.
    torch.manual_seed(11)
    rc, rt = 8, 10
    velocity = torch.randn(1, rc, rt)
    reference = torch.randn(1, rc, rt)
    out = noise_regularization(velocity.clone(), reference, lambda_kl=0.2, lambda_ac=0.0,
                               num_reg_steps=4, num_ac_rolls=5)
    print(f"    constexpr int C = {rc}, T = {rt};")
    print(carr("reg_velocity", velocity[0].numpy().ravel()))
    print(carr("reg_reference", reference[0].numpy().ravel()))
    print(carr("reg_expected", out[0].detach().numpy().ravel()))

    # --- whole solves, through audiocraft's own loop ---
    print(f"    constexpr int SC = {C}, ST = {T};")
    torch.manual_seed(5)
    prompt = torch.randn(1, C, T)
    print(carr("solve_prompt", prompt[0].numpy().ravel()))
    field = StubFlow()
    print(carr("solve_A", field.A.numpy().ravel()))
    print(carr("solve_B", field.B.numpy().ravel()))

    print("// --- guided generation, euler, no regularization (the edit's second pass) ---")
    emit_solve("solve_guided", prompt, attrs("a prompt"), solver="euler", steps=3,
               max_gen_len=T, source_flowstep=0.12, target_flowstep=1.0)

    print("// --- regularized inversion, guidance off (the edit's first pass) ---")
    n = emit_solve("solve_inverted", prompt, attrs(""), solver="euler", steps=3,
                   max_gen_len=T, source_flowstep=1.0, target_flowstep=0.12,
                   regularize=True, regularize_iters=2, keep_last_k_iters=1, lambda_kl=0.2)
    print(f"    constexpr int SOLVE_INVERTED_DRAWS = {n};")

    print("// --- the same, with regularize_iters 4 / keep_last_k 2: a real moving average ---")
    n = emit_solve("solve_inverted4", prompt, attrs(""), solver="euler", steps=3,
                   max_gen_len=T, source_flowstep=1.0, target_flowstep=0.12,
                   regularize=True, regularize_iters=4, keep_last_k_iters=2, lambda_kl=0.2)
    print(f"    constexpr int SOLVE_INVERTED4_DRAWS = {n};")

    print("// --- midpoint, guided: pins the half-step and the two-step span ---")
    emit_solve("solve_midpoint", prompt, attrs("a prompt"), solver="midpoint", steps=4,
               max_gen_len=T, source_flowstep=0.12, target_flowstep=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
