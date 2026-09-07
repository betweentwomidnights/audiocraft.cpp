#!/usr/bin/env python3
"""Dump T5-base encoder references for the ac-textenc parity check.

Runs exactly what audiocraft's `T5Conditioner.forward` runs before `output_proj`:

    inputs = t5_tokenizer(entries, return_tensors='pt', padding=True)
    embeds = t5(**inputs).last_hidden_state

so a mismatch here is a real conditioning difference, not a harness artifact. Padded
positions are zeroed, matching what the conditioner does after projecting.

Needs torch + transformers. Any of the gary4local service venvs has them, e.g.
  services/melodyflow/env/Scripts/python.exe tools/dump_t5_refs.py --prompt "..." --out refdata

Writes <out>/t5_hidden.npy as float32 [seq, dim]; with several prompts, one file per
prompt as t5_hidden.<i>.npy. Compare with tools/cossim.py.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", required=True,
                    help="repeatable; several prompts are batched and padded like audiocraft does")
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--model", default="t5-base")
    ap.add_argument("--taps", action="store_true",
                    help="also dump the per-encoder-block hidden state as t5_block<N>.npy")
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    previous = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
    finally:
        logging.disable(previous)

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5EncoderModel.from_pretrained(args.model).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=args.taps)
    hidden = out.last_hidden_state * mask.unsqueeze(-1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = hidden.shape[0]
    for i in range(n):
        name = "t5_hidden.npy" if n == 1 else f"t5_hidden.{i}.npy"
        arr = hidden[i].to(torch.float32).numpy()
        np.save(out_dir / name, arr)
        ids = inputs["input_ids"][i].tolist()
        valid = int(mask[i].sum())
        print(f"prompt {i}: {valid} valid / {len(ids)} padded token(s), "
              f"hidden {tuple(arr.shape)} -> {out_dir / name}")

    if args.taps:
        for b, h in enumerate(out.hidden_states):
            arr = (h * mask.unsqueeze(-1))[0].to(torch.float32).numpy()
            np.save(out_dir / f"t5_block{b}.npy", arr)
        print(f"wrote {len(out.hidden_states)} block taps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
