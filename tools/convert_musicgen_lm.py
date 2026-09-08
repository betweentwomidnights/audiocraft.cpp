#!/usr/bin/env python3
"""Convert a MusicGen language model (`state_dict.bin`) to GGUF.

gary loads `thepatch/*` full finetunes of musicgen small/medium/large. Those ship the raw
audiocraft pickle, not the HF `transformers` layout, so everything the model needs is read
from the embedded `xp.cfg` and nothing is hardcoded -- the same converter handles all three
sizes, and a variant it does not understand fails here rather than downstream.

What the LM is: a causal pre-norm transformer over `n_q` interleaved codebooks. Each step
sums `n_q` embedding lookups, adds a sinusoidal positional embedding, runs self-attention
(causal) and cross-attention onto projected T5-base states, and emits `n_q` sets of logits
from separate linear heads.

Three things this converter asserts rather than assumes, because the wrong value still
produces plausible audio:

  * the codebook pattern is `delay` with delays [0..n_q-1]. That interleaving is what makes
    a sequence step contain codebooks from different timesteps, and it is not recoverable
    from the weights.
  * `positional_embedding` is `sin`, not rope. The formula divides by `half_dim - 1`, not
    `half_dim` -- see `ac.lm.pos_denominator` below.
  * the fuser uses cross-attention only, with **no mask**. `T5Conditioner.forward` zeroes
    padded positions after `output_proj` (bias included) and `ConditionFuser` passes no
    mask, so padding is attended over as zero vectors. The unconditional branch is an
    all-zero context of the same length, which makes its cross-attention contribute exactly
    nothing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

import audiocraft_ckpt
import gguf_meta
from gguf_meta import ConversionError, Emitter


def read_spec(cfg):
    lm = audiocraft_ckpt.require(cfg, "transformer_lm")

    # A key absent from the config means the constructor default applies, so pass that
    # default in rather than treating "missing" as a mismatch.
    def want(key, value, default=None):
        actual = lm.get(key, default)
        if actual is None:
            actual = default
        if actual != value:
            raise ConversionError(f"unsupported transformer_lm.{key}={actual!r} "
                                  f"(this converter handles {value!r})")

    want("norm", "layer_norm")
    want("norm_first", True)
    want("positional_embedding", "sin")
    want("causal", True)
    want("activation", "gelu")
    want("kv_repeat", 1, default=1)
    want("xpos", False, default=False)
    want("two_step_cfg", False, default=False)
    if lm.get("past_context") is not None:
        raise ConversionError("past_context is not supported: attention here is over the "
                              "whole history, which is what MusicGen ships")
    if lm.get("layer_scale") is not None:
        raise ConversionError("layer_scale is not supported (this checkpoint has none)")
    if lm.get("bias_ff") or lm.get("bias_attn") or lm.get("bias_proj"):
        raise ConversionError("biased projections are not supported (this checkpoint has none)")
    if lm.get("qk_layer_norm") or lm.get("qk_layer_norm_cross"):
        raise ConversionError("qk layer norm is not supported")

    fuser = audiocraft_ckpt.require(cfg, "fuser")
    if list(fuser.get("cross") or []) != ["description"]:
        raise ConversionError(f"expected a single 'description' cross condition, got "
                              f"{fuser.get('cross')!r}")
    for op in ("sum", "prepend", "input_interpolate"):
        if fuser.get(op):
            raise ConversionError(f"fuser.{op} is not supported: {fuser[op]!r}")
    if fuser.get("cross_attention_pos_emb"):
        raise ConversionError("cross_attention_pos_emb is not supported")

    conditioner = audiocraft_ckpt.optional(cfg, "conditioners", "description", "model")
    if conditioner != "t5":
        raise ConversionError(f"expected a t5 description conditioner, got {conditioner!r}")
    t5_name = audiocraft_ckpt.optional(cfg, "conditioners", "description", "t5", "name")
    if t5_name != "t5-base":
        raise ConversionError(f"expected t5-base, got {t5_name!r}; convert_t5.py targets t5-base")

    n_q = int(lm["n_q"])
    pattern = audiocraft_ckpt.require(cfg, "codebooks_pattern")
    if pattern.get("modeling") != "delay":
        raise ConversionError(f"only the delay pattern is supported, got "
                              f"{pattern.get('modeling')!r}")
    delay = pattern.get("delay") or {}
    delays = list(delay.get("delays") or [])
    if delays != list(range(n_q)):
        raise ConversionError(f"expected delays {list(range(n_q))}, got {delays!r}")
    if delay.get("flatten_first") or delay.get("empty_initial"):
        raise ConversionError("flatten_first / empty_initial are not supported")

    dim = int(lm["dim"])
    heads = int(lm["num_heads"])
    if dim % heads:
        raise ConversionError(f"dim {dim} is not divisible by num_heads {heads}")
    return {
        "dim": dim,
        "layers": int(lm["num_layers"]),
        "heads": heads,
        "head_dim": dim // heads,
        "ff_dim": int(lm["hidden_scale"]) * dim,
        "n_q": n_q,
        "card": int(lm["card"]),
        "delays": delays,
        "max_period": float(lm.get("max_period", 10000.0)),
        "norm_eps": 1e-5,          # create_norm_fn hardcodes it
        # Unlike MelodyFlow's DiT, MusicGen's guidance coefficient really is the one in the
        # checkpoint: `MusicGen.set_generation_params` passes it explicitly at generation
        # time and gary passes 3.0, which is also both defaults.
        "cfg_coef": float(audiocraft_ckpt.optional(
            cfg, "classifier_free_guidance", "inference_coef", default=3.0)),
        "sample_rate": int(audiocraft_ckpt.optional(cfg, "sample_rate", default=32000)),
        "max_duration": float(audiocraft_ckpt.optional(
            cfg, "dataset", "segment_duration", default=30.0)),
    }


def convert(src, out, weight_type="f16", name=None, source_repo=None, revision=""):
    if weight_type not in ("f16", "f32"):
        raise ConversionError("weight_type must be f16 or f32")
    weight_dtype = np.float16 if weight_type == "f16" else np.float32
    state, cfg = audiocraft_ckpt.load(src)
    spec = read_spec(cfg)

    dim, layers, n_q, card = spec["dim"], spec["layers"], spec["n_q"], spec["card"]
    cond_dim = int(state["condition_provider.conditioners.description.output_proj.weight"].shape[1])

    # The embedding table has one extra row: `special_token_id` is `card`, the value the
    # delay pattern fills unwritten positions with. The output heads do not -- the model
    # never predicts it.
    emb_rows = int(state["emb.0.weight"].shape[0])
    if emb_rows != card + 1:
        raise ConversionError(f"embedding has {emb_rows} rows, expected card + 1 = {card + 1}")
    head_rows = int(state["linears.0.weight"].shape[0])
    if head_rows != card:
        raise ConversionError(f"output head has {head_rows} rows, expected card = {card}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="ac-musicgen-lm")
    w.add_string("ac.architecture", "audiocraft")
    w.add_uint32("ac.format_version", 1)
    w.add_string("ac.lm.family", "musicgen")
    w.add_uint32("ac.lm.dim", dim)
    w.add_uint32("ac.lm.layers", layers)
    w.add_uint32("ac.lm.heads", spec["heads"])
    w.add_uint32("ac.lm.head_dim", spec["head_dim"])
    w.add_uint32("ac.lm.ff_dim", spec["ff_dim"])
    w.add_uint32("ac.lm.cond_dim", cond_dim)
    w.add_uint32("ac.lm.n_q", n_q)
    w.add_uint32("ac.lm.card", card)
    w.add_uint32("ac.lm.special_token_id", card)
    w.add_array("ac.lm.delays", [int(d) for d in spec["delays"]])
    w.add_float32("ac.lm.max_period", spec["max_period"])
    # `create_sin_embedding` computes `pos / max_period ** (i / (half_dim - 1))`, where the
    # usual formula divides by half_dim. Recorded explicitly so the reader cannot quietly
    # pick the conventional one.
    w.add_uint32("ac.lm.pos_denominator", dim // 2 - 1)
    w.add_float32("ac.lm.norm_eps", spec["norm_eps"])
    w.add_float32("ac.lm.cfg_coef", spec["cfg_coef"])
    w.add_uint32("ac.lm.sample_rate", spec["sample_rate"])
    w.add_float32("ac.lm.max_duration", spec["max_duration"])
    w.add_string("ac.lm.activation", "gelu_erf")
    w.add_string("ac.lm.weight_type", weight_type.upper())

    e = Emitter(state, w, weight_dtype)

    for k in range(n_q):
        e.linear(f"lm.emb.{k}", f"emb.{k}")
        e.linear(f"lm.head.{k}", f"linears.{k}")
    e.norm("lm.out_norm", "out_norm")

    # T5 hidden states are projected to the model width once, before any layer sees them.
    # Unlike the rest of the model this projection has a bias.
    cond = "condition_provider.conditioners.description.output_proj"
    e.linear("lm.cond_proj", cond)
    e.put("lm.cond_proj.bias", e.take(cond + ".bias"), weight=False)

    for i in range(layers):
        src_layer = f"transformer.layers.{i}."
        dst = f"lm.{i}."
        e.norm(dst + "norm1", src_layer + "norm1")
        # Self-attention packs q, k and v in one [3*dim, dim] weight. nn.MultiheadAttention
        # names it `in_proj_weight`, not `in_proj.weight`.
        self_qkv = e.take(src_layer + "self_attn.in_proj_weight")
        if self_qkv.shape != (3 * dim, dim):
            raise ConversionError(f"self in_proj is {self_qkv.shape}, expected {(3 * dim, dim)}")
        e.put(dst + "self.qkv.weight", self_qkv)
        e.linear(dst + "self.out", src_layer + "self_attn.out_proj")
        e.norm(dst + "norm_cross", src_layer + "norm_cross")
        # Cross-attention packs them the same way, but q reads the sequence and k/v read
        # the projected T5 states, so the two halves are split here.
        cross = e.take(src_layer + "cross_attention.in_proj_weight")
        if cross.shape != (3 * dim, dim):
            raise ConversionError(f"cross in_proj is {cross.shape}, expected {(3 * dim, dim)}")
        e.put(dst + "cross.q.weight", cross[:dim])
        e.put(dst + "cross.kv.weight", cross[dim:])
        e.linear(dst + "cross.out", src_layer + "cross_attention.out_proj")
        e.norm(dst + "norm2", src_layer + "norm2")
        e.linear(dst + "ff.0", src_layer + "linear1")
        e.linear(dst + "ff.2", src_layer + "linear2")

    unexpected = e.leftovers()
    if unexpected:
        raise ConversionError(
            f"{len(unexpected)} unrecognized LM tensor(s): " + ", ".join(unexpected[:8]))

    w.add_uint64("ac.lm.parameter_count", e.params)
    label = name or out.stem
    gguf_meta.add_general(w, basename="musicgen-lm", name=label, n_params=e.params,
                          license_id="cc-by-nc-4.0")
    if source_repo:
        gguf_meta.add_source(w, label, source_repo.split("/")[0],
                             f"https://huggingface.co/{source_repo}", revision,
                             "state_dict.bin")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    spec["cond_dim"] = cond_dim
    return {"output": str(out), "tensor_count": e.count, "parameter_count": e.params,
            "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="musicgen state_dict.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-type", choices=("f16", "f32"), default="f16")
    ap.add_argument("--name", help="human-readable model name for general.name")
    ap.add_argument("--source-repo", help="e.g. thepatch/vanya_ai_dnb_0.1")
    ap.add_argument("--revision", default="", help="upstream snapshot revision")
    args = ap.parse_args()
    try:
        info = convert(args.src, args.out, args.out_type, args.name, args.source_repo,
                       args.revision)
    except (ConversionError, audiocraft_ckpt.CheckpointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    s = info["spec"]
    print(f"wrote {info['output']}")
    print(f"  {info['tensor_count']} tensors, {info['parameter_count']:,} parameters")
    print(f"  dim {s['dim']}, {s['layers']} layers, {s['heads']} heads, ff {s['ff_dim']}, "
          f"{s['n_q']} codebooks of {s['card']}, delays {s['delays']}, cfg_coef {s['cfg_coef']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
