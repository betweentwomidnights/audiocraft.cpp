#!/usr/bin/env python3
"""Convert MelodyFlow's flow-matching DiT (`state_dict.bin`) to GGUF.

`facebook/melodyflow-t24-30secs` is a 962M-parameter `FlowModel` (audiocraft
`models/flow.py`): a pre-norm transformer over 128-channel audio latents with RoPE self-
attention, cross-attention onto T5-base, U-ViT skip connections, and a timestep embedding
that is *added* to the normed input before each attention block rather than modulating it.

Several settings here are easy to get wrong because the checkpoint does not carry them and
the wrong value still produces plausible audio:

  * `cfg_coef` is 4.0, the `FlowModel.__init__` default. `get_dit_model` builds the model
    from `cfg.transformer_lm`, which has no `cfg_coef` key, so the checkpoint's
    `classifier_free_guidance.inference_coef: 3.0` is never read.
  * `add_zero_attn` prepends one all-zero key/value to *both* self- and cross-attention.
  * the feed-forward activation is `F.gelu` with `approximate='none'` -- the erf form, not
    the tanh approximation.

RoPE frequency buffers are validated against the analytic expectation and then dropped, so
a checkpoint whose rotary setup differs fails here rather than silently downstream.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

import audiocraft_ckpt
import gguf_meta

MELODYFLOW_REVISION = "77bcfce24371bf29a06152c72169"   # snapshot dir prefix; informational


class ConversionError(ValueError):
    pass


class Emitter:
    def __init__(self, state, writer, weight_dtype):
        self.state = state
        self.writer = writer
        self.weight_dtype = weight_dtype
        self.consumed = set()
        self.count = 0
        self.params = 0

    def take(self, key):
        if key not in self.state:
            raise ConversionError(f"missing tensor: {key}")
        self.consumed.add(key)
        return self.state[key]

    def put(self, name, array, weight=True):
        # Norms, biases and the latent statistics stay F32: they are a rounding error's
        # worth of file size and the places where half precision actually shows.
        a = np.ascontiguousarray(array.astype(self.weight_dtype if weight else np.float32))
        self.writer.add_tensor(name, a)
        self.count += 1
        self.params += int(a.size)

    def linear(self, dst, src, weight=True):
        self.put(dst + ".weight", self.take(src + ".weight"), weight)

    def norm(self, dst, src):
        self.put(dst + ".weight", self.take(src + ".weight"), weight=False)
        self.put(dst + ".bias", self.take(src + ".bias"), weight=False)


def check_rope_frequencies(freqs, head_dim, max_period):
    """Validate the stored rotary frequencies -- but keep them, do not recompute.

    `RotaryEmbedding` registers `1 / max_period ** (arange(0, dim, 2) / dim)`. In this
    checkpoint that buffer has been round-tripped through bfloat16 at some point (the
    weights have not): the stored values are exactly `bfloat16(analytic)`, so e.g. index 1
    is 0.75 rather than 0.74989421, a relative error up to 3.7e-3.

    That is not negligible. The angle at a given position is `pos * freq`, and MelodyFlow's
    window runs to position 749, so a 1.4e-4 relative error on the second frequency is
    already 0.08 radians of rotation by the end of the sequence. torch loads these exact
    values and rotates with them, so matching the reference means using them, not the
    closed form.

    Hence the loose tolerance: this catches a genuinely different rotary setup (a changed
    base, a changed head dim) while accepting the half-precision rounding that is actually
    in the file.
    """
    expected = 1.0 / (max_period ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    if freqs.shape != expected.shape:
        raise ConversionError(
            f"rope frequency buffer is {freqs.shape}, expected {expected.shape} "
            f"for head_dim {head_dim}")
    if not np.allclose(freqs.astype(np.float64), expected, rtol=0.01, atol=1e-8):
        raise ConversionError(
            "rope frequencies are not a rounded 1 / max_period ** (arange(0, dim, 2) / dim) "
            f"with max_period {max_period}; this checkpoint uses a different rotary setup")


def rope_freq_factors(freqs):
    """Turn stored frequencies into ggml_rope_ext's `freq_factors`.

    ggml computes `theta_i = (pos * freq_base ** (-2i/n_dims)) / freq_factors[i]`. Passing
    `freq_base = 1` collapses the first factor to 1 for every i, so `theta_i = pos /
    freq_factors[i]` and `freq_factors[i] = 1 / freq[i]` reproduces the stored frequencies
    exactly -- with no dependence on how ggml accumulates its own power series.
    """
    f = freqs.astype(np.float64)
    if np.any(f <= 0.0):
        raise ConversionError("rope frequencies must be positive")
    return np.ascontiguousarray((1.0 / f).astype(np.float32))


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
    want("positional_embedding", "rope")
    want("cross_attention", True)
    want("skip_connections", True)
    want("add_zero_attn", True)
    want("activation", "gelu")
    want("causal", False, default=False)
    want("kv_repeat", 1, default=1)
    want("xpos", False, default=False)
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

    conditioner = audiocraft_ckpt.optional(cfg, "conditioners", "description", "model")
    if conditioner != "t5":
        raise ConversionError(f"expected a t5 description conditioner, got {conditioner!r}")
    t5_name = audiocraft_ckpt.optional(cfg, "conditioners", "description", "t5", "name")
    if t5_name != "t5-base":
        raise ConversionError(f"expected t5-base, got {t5_name!r}; convert_t5.py targets t5-base")

    dim = int(lm["dim"])
    heads = int(lm["num_heads"])
    if dim % heads:
        raise ConversionError(f"dim {dim} is not divisible by num_heads {heads}")
    spec = {
        "dim": dim,
        "layers": int(lm["num_layers"]),
        "heads": heads,
        "head_dim": dim // heads,
        "ff_dim": int(lm["hidden_scale"]) * dim,
        "max_period": float(lm.get("max_period", 10000.0)),
        "norm_eps": 1e-5,          # create_norm_fn hardcodes it
        # FlowModel.__init__'s default. See the module docstring: the checkpoint's
        # classifier_free_guidance.inference_coef is NOT what the model uses.
        "cfg_coef": float(lm.get("cfg_coef", 4.0)),
        "max_duration": float(audiocraft_ckpt.optional(
            cfg, "dataset", "segment_duration", default=30.0)),
    }
    spec["n_skip"] = (spec["layers"] - 1) // 2
    return spec


def convert(src, out, weight_type="f16"):
    if weight_type not in ("f16", "f32"):
        raise ConversionError("weight_type must be f16 or f32")
    weight_dtype = np.float16 if weight_type == "f16" else np.float32
    state, cfg = audiocraft_ckpt.load(src)
    spec = read_spec(cfg)

    dim, layers, head_dim = spec["dim"], spec["layers"], spec["head_dim"]
    latent_dim = int(state["latent_mean"].reshape(-1).shape[0])
    time_dim = int(state["timestep_embedder.mlp.0.weight"].shape[1])
    cond_dim = int(state["condition_provider.conditioners.description.output_proj.weight"].shape[1])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="ac-melodyflow-dit")
    w.add_string("ac.architecture", "audiocraft")
    w.add_uint32("ac.format_version", 1)
    w.add_string("ac.dit.family", "melodyflow")
    w.add_uint32("ac.dit.dim", dim)
    w.add_uint32("ac.dit.layers", layers)
    w.add_uint32("ac.dit.heads", spec["heads"])
    w.add_uint32("ac.dit.head_dim", head_dim)
    w.add_uint32("ac.dit.ff_dim", spec["ff_dim"])
    w.add_uint32("ac.dit.latent_dim", latent_dim)
    w.add_uint32("ac.dit.cond_dim", cond_dim)
    w.add_uint32("ac.dit.time_dim", time_dim)
    w.add_uint32("ac.dit.n_skip", spec["n_skip"])
    w.add_float32("ac.dit.rope_base", spec["max_period"])
    w.add_float32("ac.dit.norm_eps", spec["norm_eps"])
    w.add_float32("ac.dit.cfg_coef", spec["cfg_coef"])
    w.add_float32("ac.dit.max_duration", spec["max_duration"])
    w.add_bool("ac.dit.add_zero_attn", True)
    w.add_string("ac.dit.activation", "gelu_erf")
    w.add_string("ac.dit.weight_type", weight_type.upper())

    e = Emitter(state, w, weight_dtype)

    # Every layer shares one RotaryEmbedding, so the state dict repeats the same buffer.
    # Validate all of them, insist they agree, and emit one.
    shared_freqs = e.take("transformer.rope.frequencies")
    check_rope_frequencies(shared_freqs, head_dim, spec["max_period"])
    for i in range(layers):
        per_layer = e.take(f"transformer.layers.{i}.self_attn.rope.frequencies")
        check_rope_frequencies(per_layer, head_dim, spec["max_period"])
        if not np.array_equal(per_layer, shared_freqs):
            raise ConversionError(
                f"layer {i} has different rope frequencies from the shared buffer; this "
                "converter emits one table for the whole stack")
    e.put("dit.rope_freq_factors", rope_freq_factors(shared_freqs), weight=False)

    e.linear("dit.in_proj", "in_proj")
    e.linear("dit.out_proj", "out_proj")
    e.norm("dit.out_norm", "out_norm")

    # TimestepEmbedding: Linear(time_dim, dim) -> SiLU -> Linear(dim, dim), no biases.
    e.linear("dit.time_embed.0", "timestep_embedder.mlp.0")
    e.linear("dit.time_embed.2", "timestep_embedder.mlp.2")

    # T5 hidden states are projected to the model width once, before any layer sees them.
    cond = "condition_provider.conditioners.description.output_proj"
    e.linear("dit.cond_proj", cond)
    e.put("dit.cond_proj.bias", e.take(cond + ".bias"), weight=False)

    for i in range(layers):
        src_layer = f"transformer.layers.{i}."
        dst = f"dit.{i}."
        e.norm(dst + "norm1", src_layer + "norm1")
        # Self-attention packs q, k and v in one [3*dim, dim] weight. nn.MultiheadAttention
        # names it `in_proj_weight`, not `in_proj.weight`.
        self_qkv = e.take(src_layer + "self_attn.in_proj_weight")
        if self_qkv.shape != (3 * dim, dim):
            raise ConversionError(f"self in_proj is {self_qkv.shape}, expected {(3 * dim, dim)}")
        e.put(dst + "self.qkv.weight", self_qkv)
        e.linear(dst + "self.out", src_layer + "self_attn.out_proj")
        e.norm(dst + "norm_cross", src_layer + "norm_cross")
        # Cross-attention packs them the same way, but q reads the latents and k/v read
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

    for i in range(spec["n_skip"]):
        e.linear(f"dit.skip.{i}", f"transformer.skip_projections.{i}")

    # Latent statistics tracked during codec training; generation de-normalizes with them.
    e.put("dit.latent_mean", e.take("latent_mean").reshape(-1), weight=False)
    e.put("dit.latent_std", e.take("latent_std").reshape(-1), weight=False)

    unexpected = sorted(set(state) - e.consumed)
    if unexpected:
        raise ConversionError(
            f"{len(unexpected)} unrecognized DiT tensor(s): " + ", ".join(unexpected[:8]))

    w.add_uint64("ac.dit.parameter_count", e.params)
    gguf_meta.add_general(w, basename="melodyflow-dit", name="MelodyFlow T24 DiT",
                          n_params=e.params, license_id="cc-by-nc-4.0")
    gguf_meta.add_source(w, "MelodyFlow T24 30secs", "Meta",
                         "https://huggingface.co/facebook/melodyflow-t24-30secs",
                         MELODYFLOW_REVISION, "state_dict.bin")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    spec.update({"latent_dim": latent_dim, "cond_dim": cond_dim, "time_dim": time_dim})
    return {"output": str(out), "tensor_count": e.count, "parameter_count": e.params,
            "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="melodyflow state_dict.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-type", choices=("f16", "f32"), default="f16")
    a = ap.parse_args()
    try:
        r = convert(a.src, a.out, a.out_type)
    except (ConversionError, audiocraft_ckpt.CheckpointError, OSError, KeyError) as exc:
        sys.exit(f"error: {exc}")
    s = r["spec"]
    print(f"wrote {r['output']} ({r['tensor_count']} tensors, "
          f"{r['parameter_count']:,} parameters, {a.out_type.upper()})")
    print(f"  dim {s['dim']}, {s['layers']} layers, {s['heads']} heads x {s['head_dim']}, "
          f"ff {s['ff_dim']}, latent {s['latent_dim']}, cond {s['cond_dim']}, "
          f"{s['n_skip']} skips, cfg_coef {s['cfg_coef']}")


if __name__ == "__main__":
    main()
