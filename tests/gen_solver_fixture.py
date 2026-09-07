"""Regenerate the fixtures embedded in tests/solver_test.cpp.

Run with terry's venv so `audiocraft.utils.renoise` is importable:
  PYTHONPATH=.../services/melodyflow .../env/Scripts/python.exe tests/gen_solver_fixture.py
"""
import math
import numpy as np
import torch
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


print("// --- schedules: terry's inversion (1.0 -> 0.12) and generation (0.12 -> 1.0) ---")
print(carr("inversion_schedule", schedule(25, 1.0, 0.12).tolist()))
print(carr("generation_schedule", schedule(25, 0.12, 1.0).tolist()))
print("// --- a uniform schedule, sway 0, to pin the mapping without the warp ---")
print(carr("uniform_schedule", schedule(4, 0.0, 1.0, sway=0.0).tolist()))

# noise_regularization over a small but structurally faithful case: channels a multiple of
# 4, frames deliberately NOT a multiple of 4 so the dropped tail is covered.
torch.manual_seed(11)
C, T = 8, 10
velocity = torch.randn(1, C, T)
reference = torch.randn(1, C, T)
out = noise_regularization(velocity.clone(), reference, lambda_kl=0.2, lambda_ac=0.0,
                           num_reg_steps=4, num_ac_rolls=5)
print(f"    constexpr int C = {C}, T = {T};")
print(carr("reg_velocity", velocity[0].numpy().ravel()))
print(carr("reg_reference", reference[0].numpy().ravel()))
print(carr("reg_expected", out[0].detach().numpy().ravel()))
