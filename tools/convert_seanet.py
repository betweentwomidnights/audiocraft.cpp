#!/usr/bin/env python3
"""Convert an audiocraft SEANet codec (`compression_state_dict.bin`) to GGUF.

One converter serves both codecs this repo needs, because they are the same architecture
with different settings:

  MelodyFlow  48 kHz stereo, ratios [8,8,6,5] -> 25 Hz, snake activation, quantizer
              `no_quant` (a continuous VAE: the encoder emits mean||scale)
  MusicGen    32 kHz mono,   ratios [8,5,4,4] -> 50 Hz, ELU activation, RVQ 4x2048

Both put a 2-layer LSTM in the bottleneck and use reflect padding.

Everything constant is folded here so the ggml graph stays a plain composition of stock
operations:
  * weight_norm is fused (`g * v / ||v||`), matching what MelodyFlow.get_pretrained does
    at load time via `remove_parametrizations`;
  * snake's `1 / (alpha + 1e-9)` reciprocal is precomputed;
  * ConvTranspose1d weights are written in the col2im layout;
  * the LSTM's two bias vectors are summed (torch adds b_ih and b_hh unconditionally).

RVQ conversion is not implemented yet -- see docs/ROADMAP.md phase 5.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

import audiocraft_ckpt
import gguf_meta


class ConversionError(ValueError):
    pass


def fuse_weight_norm(g, v):
    """PyTorch weight_norm(dim=0): g * v / norm(v, dims=1..N).

    Ported from sa3.cpp tools/convert_sat_oobleck.py. This is correct for
    ConvTranspose1d too: its weight is [in, out, k] and dim=0 normalizes per *input*
    channel, which is exactly what the stored g shape [in, 1, 1] says.
    """
    if v.ndim < 2 or g.shape[0] != v.shape[0]:
        raise ConversionError(f"invalid weight_norm shapes: g={g.shape}, v={v.shape}")
    axes = tuple(range(1, v.ndim))
    vf = v.astype(np.float64)
    norm = np.sqrt(np.sum(vf * vf, axis=axes, keepdims=True))
    if np.any(norm == 0):
        raise ConversionError("cannot fuse a zero-norm weight_v slice")
    return np.ascontiguousarray((g.astype(np.float64) * vf / norm).astype(np.float32))


def conv_transpose_for_col2im(fused):
    """[in, out, kernel] -> numpy [out*kernel, in] -> GGML [in, kernel*out]."""
    if fused.ndim != 3:
        raise ConversionError(f"ConvTranspose1d weight must be rank 3, got {fused.shape}")
    return np.ascontiguousarray(fused.reshape(fused.shape[0], -1).T)


class Emitter:
    """Collects tensors and tracks which source keys were consumed."""

    def __init__(self, state, writer):
        self.state = state
        self.writer = writer
        self.consumed = set()
        self.count = 0
        self.params = 0

    def take(self, key):
        if key not in self.state:
            raise ConversionError(f"missing tensor: {key}")
        self.consumed.add(key)
        return self.state[key]

    def put(self, name, array):
        a = np.ascontiguousarray(array.astype(np.float32))
        self.writer.add_tensor(name, a)
        self.count += 1
        self.params += int(a.size)

    def conv(self, dst, src):
        """A weight-normed StreamableConv1d: torch [out, in, k] -> ggml [k, in, out]."""
        w = fuse_weight_norm(
            self.take(src + ".conv.conv.parametrizations.weight.original0"),
            self.take(src + ".conv.conv.parametrizations.weight.original1"))
        self.put(dst + ".weight", w)
        self.put(dst + ".bias", self.take(src + ".conv.conv.bias"))

    def conv_transpose(self, dst, src):
        w = fuse_weight_norm(
            self.take(src + ".convtr.convtr.parametrizations.weight.original0"),
            self.take(src + ".convtr.convtr.parametrizations.weight.original1"))
        self.put(dst + ".weight_col2im", conv_transpose_for_col2im(w))
        self.put(dst + ".bias", self.take(src + ".convtr.convtr.bias"))

    def snake(self, dst, src):
        """Snake1d: x + (alpha + 1e-9).reciprocal() * sin(alpha * x)**2.

        alpha is stored as [1, C, 1]; we write [C] plus the precomputed reciprocal so the
        graph never recomputes a constant.
        """
        alpha = self.take(src + ".alpha").reshape(-1).astype(np.float64)
        self.put(dst + ".alpha", alpha)
        self.put(dst + ".alpha_recip", 1.0 / (alpha + 1e-9))

    def elu(self, dst, src):
        """ELU is parameter-free; nothing to emit. Present for symmetry with snake()."""
        return

    def lstm(self, dst, src, layers):
        for layer in range(layers):
            p = f"{dst}.{layer}."
            s = f"{src}.lstm."
            self.put(p + "w_ih", self.take(s + f"weight_ih_l{layer}"))
            self.put(p + "w_hh", self.take(s + f"weight_hh_l{layer}"))
            # torch adds both bias vectors every step; one fused vector is equivalent.
            self.put(p + "bias", self.take(s + f"bias_ih_l{layer}")
                     + self.take(s + f"bias_hh_l{layer}"))


def read_spec(cfg):
    seanet = audiocraft_ckpt.require(cfg, "seanet")
    encodec = audiocraft_ckpt.require(cfg, "encodec")

    autoencoder = encodec.get("autoencoder", "seanet")
    if autoencoder != "seanet":
        raise ConversionError(f"unsupported autoencoder {autoencoder!r} (expected 'seanet')")

    quantizer = encodec.get("quantizer", "no_quant")
    if quantizer not in ("no_quant", "rvq"):
        raise ConversionError(f"unrecognized quantizer {quantizer!r}")
    if quantizer == "rvq":
        raise ConversionError(
            "RVQ codecs (MusicGen's EnCodec 32 kHz) are not implemented yet; "
            "see docs/ROADMAP.md phase 5")

    activation = seanet.get("activation", "ELU")
    if activation not in ("snake", "ELU"):
        raise ConversionError(f"unsupported activation {activation!r} (expected snake or ELU)")
    if seanet.get("norm", "none") != "weight_norm":
        raise ConversionError(
            f"expected norm='weight_norm', got {seanet.get('norm')!r}; this converter fuses "
            "weight norm and has nothing to do for other normalizations")
    if seanet.get("pad_mode", "reflect") != "reflect":
        raise ConversionError(f"unsupported pad_mode {seanet.get('pad_mode')!r}")
    if seanet.get("causal", False):
        raise ConversionError("causal SEANet is not supported (neither target model is causal)")
    if not seanet.get("true_skip", True):
        raise ConversionError("true_skip=False (convolutional shortcut) is not supported")
    if seanet.get("disable_norm_outer_blocks", 0):
        raise ConversionError("disable_norm_outer_blocks is not supported")
    final_activation = audiocraft_ckpt.optional(seanet, "decoder", "final_activation")
    if final_activation is not None:
        raise ConversionError(f"unsupported decoder final_activation {final_activation!r}")

    ratios = [int(r) for r in seanet["ratios"]]  # decoder order
    spec = {
        "quantizer": quantizer,
        "activation": activation,
        "channels": int(seanet["channels"]),
        "n_filters": int(seanet["n_filters"]),
        "n_residual_layers": int(seanet["n_residual_layers"]),
        "ratios": ratios,
        "kernel_size": int(seanet.get("kernel_size", 7)),
        "residual_kernel_size": int(seanet.get("residual_kernel_size", 3)),
        "last_kernel_size": int(seanet.get("last_kernel_size", 7)),
        "dilation_base": int(seanet.get("dilation_base", 2)),
        "compress": int(seanet.get("compress", 2)),
        "lstm_layers": int(seanet.get("lstm", 0)),
        "latent_dim": int(seanet["dimension"]),
        "sample_rate": int(encodec["sample_rate"]),
    }
    # The encoder may override `dimension`; a VAE does, emitting mean||scale.
    spec["encoder_dim"] = int(audiocraft_ckpt.optional(
        seanet, "encoder", "dimension", default=spec["latent_dim"]))
    spec["hop_length"] = int(np.prod(ratios))
    if spec["sample_rate"] % spec["hop_length"]:
        raise ConversionError(
            f"sample_rate {spec['sample_rate']} is not a multiple of the hop "
            f"{spec['hop_length']}")
    spec["frame_rate"] = spec["sample_rate"] // spec["hop_length"]
    if int(encodec["channels"]) != spec["channels"]:
        raise ConversionError("encodec.channels disagrees with seanet.channels")
    if quantizer == "no_quant" and spec["encoder_dim"] != 2 * spec["latent_dim"]:
        raise ConversionError(
            f"quantizer-free codec should emit mean||scale: encoder dim "
            f"{spec['encoder_dim']} is not 2 x latent dim {spec['latent_dim']}")
    return spec


def convert(src, out, weight_type="f32"):
    if weight_type != "f32":
        # The codec is ~57M parameters; there is little to gain by shrinking it, and the
        # conv stack is the part of the pipeline most sensitive to precision.
        raise ConversionError("only f32 is supported for the codec")
    state, cfg = audiocraft_ckpt.load(src)
    spec = read_spec(cfg)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="ac-seanet")
    w.add_string("ac.architecture", "audiocraft")
    w.add_uint32("ac.format_version", 1)
    w.add_string("ac.codec.quantizer", spec["quantizer"])
    w.add_string("ac.codec.activation", spec["activation"])
    w.add_uint32("ac.codec.channels", spec["channels"])
    w.add_uint32("ac.codec.n_filters", spec["n_filters"])
    w.add_uint32("ac.codec.n_residual_layers", spec["n_residual_layers"])
    w.add_array("ac.codec.ratios", [int(r) for r in spec["ratios"]])
    w.add_uint32("ac.codec.kernel_size", spec["kernel_size"])
    w.add_uint32("ac.codec.residual_kernel_size", spec["residual_kernel_size"])
    w.add_uint32("ac.codec.last_kernel_size", spec["last_kernel_size"])
    w.add_uint32("ac.codec.dilation_base", spec["dilation_base"])
    w.add_uint32("ac.codec.compress", spec["compress"])
    w.add_uint32("ac.codec.lstm_layers", spec["lstm_layers"])
    w.add_uint32("ac.codec.latent_dim", spec["latent_dim"])
    w.add_uint32("ac.codec.encoder_dim", spec["encoder_dim"])
    w.add_uint32("ac.codec.sample_rate", spec["sample_rate"])
    w.add_uint32("ac.codec.hop_length", spec["hop_length"])
    w.add_uint32("ac.codec.frame_rate", spec["frame_rate"])
    w.add_string("ac.codec.weight_type", weight_type.upper())

    e = Emitter(state, w)
    act = e.snake if spec["activation"] == "snake" else e.elu
    n_res = spec["n_residual_layers"]
    lstm_layers = spec["lstm_layers"]

    def residual(dst, src_prefix):
        act(dst + ".snake1", src_prefix + ".block.0")
        e.conv(dst + ".conv1", src_prefix + ".block.1")
        act(dst + ".snake2", src_prefix + ".block.2")
        e.conv(dst + ".conv2", src_prefix + ".block.3")

    # ---- encoder: conv, then (residuals, act, downsample) per reversed ratio, then
    # ---- LSTM, act, conv.
    idx = 0
    e.conv("codec.encoder.in", f"encoder.model.{idx}")
    for stage in range(len(spec["ratios"])):
        for j in range(n_res):
            idx += 1
            residual(f"codec.encoder.stage.{stage}.res.{j}", f"encoder.model.{idx}")
        idx += 1
        act(f"codec.encoder.stage.{stage}.snake", f"encoder.model.{idx}")
        idx += 1
        e.conv(f"codec.encoder.stage.{stage}.down", f"encoder.model.{idx}")
    if lstm_layers:
        idx += 1
        e.lstm("codec.encoder.lstm", f"encoder.model.{idx}", lstm_layers)
    idx += 1
    act("codec.encoder.out_snake", f"encoder.model.{idx}")
    idx += 1
    e.conv("codec.encoder.out", f"encoder.model.{idx}")

    # ---- decoder: conv, LSTM, then (act, upsample, residuals) per ratio, act, conv.
    idx = 0
    e.conv("codec.decoder.in", f"decoder.model.{idx}")
    if lstm_layers:
        idx += 1
        e.lstm("codec.decoder.lstm", f"decoder.model.{idx}", lstm_layers)
    for stage in range(len(spec["ratios"])):
        idx += 1
        act(f"codec.decoder.stage.{stage}.snake", f"decoder.model.{idx}")
        idx += 1
        e.conv_transpose(f"codec.decoder.stage.{stage}.up", f"decoder.model.{idx}")
        for j in range(n_res):
            idx += 1
            residual(f"codec.decoder.stage.{stage}.res.{j}", f"decoder.model.{idx}")
    idx += 1
    act("codec.decoder.out_snake", f"decoder.model.{idx}")
    idx += 1
    e.conv("codec.decoder.out", f"decoder.model.{idx}")

    unexpected = sorted(set(state) - e.consumed)
    if unexpected:
        raise ConversionError(
            f"{len(unexpected)} unrecognized codec tensor(s): " + ", ".join(unexpected[:8]))

    w.add_uint64("ac.codec.parameter_count", e.params)
    basename = "melodyflow-vae" if spec["quantizer"] == "no_quant" else "encodec"
    gguf_meta.add_general(w, basename=basename, name="audiocraft SEANet codec",
                          n_params=e.params, license_id="cc-by-nc-4.0")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"output": str(out), "tensor_count": e.count, "parameter_count": e.params,
            "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="audiocraft compression_state_dict.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-type", choices=("f32",), default="f32")
    a = ap.parse_args()
    try:
        r = convert(a.src, a.out, a.out_type)
    except (ConversionError, audiocraft_ckpt.CheckpointError, OSError, KeyError) as exc:
        sys.exit(f"error: {exc}")
    s = r["spec"]
    print(f"wrote {r['output']} ({r['tensor_count']} tensors, "
          f"{r['parameter_count']:,} parameters)")
    print(f"  {s['sample_rate']} Hz, {s['channels']}ch, ratios {s['ratios']} "
          f"-> hop {s['hop_length']} / {s['frame_rate']} Hz, {s['activation']}, "
          f"lstm {s['lstm_layers']}, latent {s['latent_dim']} "
          f"(encoder emits {s['encoder_dim']}), quantizer {s['quantizer']}")


if __name__ == "__main__":
    main()
