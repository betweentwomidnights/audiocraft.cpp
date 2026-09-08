"""Regenerate the fixtures embedded in tests/pattern_test.cpp.

Run with gary's venv so `audiocraft` is importable:
  PYTHONPATH=.../services/gary .../env/Scripts/python.exe tests/gen_pattern_fixture.py

The fixtures come from the real `DelayedPatternProvider`, driven through the same
`build_pattern_sequence` / `revert_pattern_sequence` calls `LMModel.generate` makes -- so
the C++ closed form is checked against audiocraft's layout-and-scatter-index construction
rather than against a second transcription of it.
"""
import sys

import torch
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider


def carr(name, values, kind="int32_t"):
    lines, cur = [], "        "
    vals = list(values)
    for i, v in enumerate(vals):
        piece = f"{int(v)}" + ("," if i + 1 < len(vals) else "")
        if len(cur) + len(piece) > 92:
            lines.append(cur.rstrip()); cur = "        "
        cur += piece + " "
    lines.append(cur.rstrip())
    return f"    static const {kind} {name}[] = {{\n" + "\n".join(lines) + "\n    };"


def emit(tag, n_q, timesteps, special):
    provider = DelayedPatternProvider(n_q=n_q)
    pattern = provider.get_pattern(timesteps)
    # Distinct, recognisable codes: codebook q at timestep t is 100*q + t, so a swapped
    # axis or an off-by-one delay is obvious in the failure message.
    codes = torch.arange(timesteps).view(1, 1, -1).repeat(1, n_q, 1)
    codes = codes + torch.arange(n_q).view(1, -1, 1) * 100
    seq, _, mask = pattern.build_pattern_sequence(codes, special_token=special)
    back, _, _ = pattern.revert_pattern_sequence(seq, special_token=special)

    steps = seq.shape[-1]
    print(f"    // {tag}: n_q={n_q}, timesteps={timesteps}, special={special}")
    print(f"    constexpr int {tag}_NQ = {n_q}, {tag}_T = {timesteps}, "
          f"{tag}_S = {steps}, {tag}_SPECIAL = {special};")
    print(carr(f"{tag}_codes", codes[0].reshape(-1).tolist()))
    print(carr(f"{tag}_sequence", seq[0].reshape(-1).tolist()))
    print(carr(f"{tag}_mask", mask.reshape(-1).to(torch.int32).tolist(), kind="int32_t"))
    print(carr(f"{tag}_reverted", back[0].reshape(-1).tolist()))
    first = [pattern.get_first_step_with_timesteps(t) for t in range(timesteps)]
    print(carr(f"{tag}_first_step", first))


def main():
    # The shipped model's shape, short enough to read.
    emit("MG", 4, 6, 2048)
    # A single codebook degenerates to no interleaving at all.
    emit("ONE", 1, 5, 2048)
    # An odd count, and a length shorter than the deepest delay so the "audio ends before the
    # sequence does" tail is exercised from both directions.
    emit("THREE", 3, 2, 2048)
    return 0


if __name__ == "__main__":
    sys.exit(main())
