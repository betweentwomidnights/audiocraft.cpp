"""Regenerate the fixtures embedded in tests/sampling_test.cpp.

Run with gary's venv so `audiocraft.utils.utils` is importable:
  PYTHONPATH=.../services/gary .../env/Scripts/python.exe tests/gen_sampling_fixture.py

Everything about top-k sampling except the draw itself is deterministic: the softmax, the
cutoff, which tokens survive it, and the renormalized distribution. Those are what get
pinned here, against `audiocraft.utils.utils.sample_top_k`'s own arithmetic. The draw is
not comparable -- torch's RNG stream is not ours -- and is checked for its own properties in
the test instead.

The fixture deliberately includes a deliberate tie at the cutoff, because the reference
compares with `>=` and therefore keeps *every* token matching the k-th value, which can
leave more than k candidates.
"""
import sys

import torch


def carr(name, values):
    lines, cur = [], "        "
    vals = list(values)
    for i, v in enumerate(vals):
        text = f"{float(v):.9g}"
        if "." not in text and "e" not in text and "n" not in text:
            text += ".0"
        piece = text + "f" + ("," if i + 1 < len(vals) else "")
        if len(cur) + len(piece) > 92:
            lines.append(cur.rstrip()); cur = "        "
        cur += piece + " "
    lines.append(cur.rstrip())
    return f"    static const float {name}[] = {{\n" + "\n".join(lines) + "\n    };"


def emit(tag, logits, k, temperature):
    """`sample_top_k` up to the multinomial draw, transcribed from audiocraft."""
    probs = torch.softmax(logits / temperature, dim=-1)
    filtered = probs.clone()
    if 0 < k < probs.shape[-1]:
        top_k_value, _ = torch.topk(filtered, k, dim=-1)
        min_value_top_k = top_k_value[..., [-1]]
        filtered = filtered * (filtered >= min_value_top_k).float()
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    survivors = int((filtered > 0).sum())
    print(f"    // {tag}: n={logits.shape[-1]}, k={k}, temperature={temperature}, "
          f"{survivors} survivor(s)")
    print(f"    constexpr int {tag}_N = {logits.shape[-1]}, {tag}_K = {k}, "
          f"{tag}_SURVIVORS = {survivors};")
    print(f"    constexpr float {tag}_TEMPERATURE = {temperature}f;")
    print(carr(f"{tag}_logits", logits.tolist()))
    print(carr(f"{tag}_softmax", probs.tolist()))
    print(carr(f"{tag}_filtered", filtered.tolist()))
    print(f"    constexpr int {tag}_ARGMAX = {int(torch.argmax(logits))};")


def main():
    torch.manual_seed(17)
    # Small enough to read, wide enough that top-k really discards most of it.
    logits = torch.randn(24) * 3.0
    emit("PLAIN", logits, k=6, temperature=1.0)
    emit("HOT", logits, k=6, temperature=2.5)
    emit("COLD", logits, k=6, temperature=0.4)
    emit("WIDE", logits, k=24, temperature=1.0)      # k == n: nothing is discarded

    # A tie exactly at the cutoff: the reference's `>=` keeps both, so more than k survive.
    tied = logits.clone()
    order = torch.argsort(tied, descending=True)
    tied[order[6]] = tied[order[5]]
    emit("TIED", tied, k=6, temperature=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
