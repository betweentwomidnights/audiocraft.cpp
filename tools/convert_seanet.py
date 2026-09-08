#!/usr/bin/env python3
"""Convert a SEANet codec to GGUF.

One converter serves both codecs this repo needs, because they are the same architecture
with different settings:

  MelodyFlow  48 kHz stereo, ratios [8,8,6,5] -> 25 Hz, snake activation, quantizer
              `no_quant` (a continuous VAE: the encoder emits mean||scale)
  MusicGen    32 kHz mono,   ratios [8,5,4,4] -> 50 Hz, ELU activation, RVQ 4x2048

Both put a 2-layer LSTM in the bottleneck and use reflect padding.

They arrive in **two different file formats**, which is the only reason this file has a
notion of dialect. MelodyFlow ships audiocraft's own `compression_state_dict.bin`;
MusicGen's `compression_state_dict.bin` merely says `pretrained: facebook/encodec_32khz`,
and audiocraft then loads that through HuggingFace `transformers` as an `EncodecModel`.
That is a faithful port of the same SEANet -- same layer order, same padding arithmetic,
same residual blocks -- but the tensors are named differently and weight norm is stored in
the legacy `weight_g`/`weight_v` form rather than as torch parametrizations.

The layer *indices* are identical in both, which is worth stating because it looks like a
coincidence and is not: activations occupy a module slot without owning parameters, so both
encoders run 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15.

Everything constant is folded here so the ggml graph stays a plain composition of stock
operations:
  * weight_norm is fused (`g * v / ||v||`), matching what MelodyFlow.get_pretrained does
    at load time via `remove_parametrizations`;
  * snake's `1 / (alpha + 1e-9)` reciprocal is precomputed;
  * ConvTranspose1d weights are written in the col2im layout;
  * the LSTM's two bias vectors are summed (torch adds b_ih and b_hh unconditionally).

"""

import argparse
import sys
from pathlib import Path

import json

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


class AudiocraftDialect:
    """audiocraft's own `compression_state_dict.bin` (MelodyFlow).

    Weight norm is stored as torch parametrizations, and the module lists are `.model`.
    """
    name = "audiocraft"
    encoder = "encoder.model.{}"
    decoder = "decoder.model.{}"

    def conv(self, src):
        b = src + ".conv.conv."
        return b + "parametrizations.weight.original0", b + "parametrizations.weight.original1", b + "bias"

    def conv_transpose(self, src):
        b = src + ".convtr.convtr."
        return b + "parametrizations.weight.original0", b + "parametrizations.weight.original1", b + "bias"

    def lstm(self, src, layer):
        b = src + ".lstm."
        return (b + f"weight_ih_l{layer}", b + f"weight_hh_l{layer}",
                b + f"bias_ih_l{layer}", b + f"bias_hh_l{layer}")


class HuggingFaceDialect:
    """`transformers.EncodecModel` (MusicGen's facebook/encodec_32khz).

    Legacy `weight_g`/`weight_v` rather than parametrizations, and `.layers` rather than
    `.model`. Everything else -- indices, block layout, padding -- matches.
    """
    name = "huggingface"
    encoder = "encoder.layers.{}"
    decoder = "decoder.layers.{}"

    def conv(self, src):
        b = src + ".conv."
        return b + "weight_g", b + "weight_v", b + "bias"

    conv_transpose = conv

    def lstm(self, src, layer):
        b = src + ".lstm."
        return (b + f"weight_ih_l{layer}", b + f"weight_hh_l{layer}",
                b + f"bias_ih_l{layer}", b + f"bias_hh_l{layer}")

    def codebook(self, index):
        return f"quantizer.layers.{index}.codebook.embed"


class Emitter:
    """Collects tensors and tracks which source keys were consumed."""

    def __init__(self, state, writer, dialect):
        self.dialect = dialect
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
        g, v, bias = self.dialect.conv(src)
        self.put(dst + ".weight", fuse_weight_norm(self.take(g), self.take(v)))
        self.put(dst + ".bias", self.take(bias))

    def conv_transpose(self, dst, src):
        g, v, bias = self.dialect.conv_transpose(src)
        w = fuse_weight_norm(self.take(g), self.take(v))
        self.put(dst + ".weight_col2im", conv_transpose_for_col2im(w))
        self.put(dst + ".bias", self.take(bias))

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
            w_ih, w_hh, b_ih, b_hh = self.dialect.lstm(src, layer)
            self.put(p + "w_ih", self.take(w_ih))
            self.put(p + "w_hh", self.take(w_hh))
            # torch adds both bias vectors every step; one fused vector is equivalent.
            self.put(p + "bias", self.take(b_ih) + self.take(b_hh))

    def codebook(self, index):
        """One RVQ codebook, [bins, dim] -> ggml [dim, bins].

        Emitted as-is: decoding is a row lookup and encoding is a nearest-neighbour search,
        and both want the vectors contiguous. The EMA state that sits alongside it
        (`cluster_size`, `embed_avg`, `inited`) is training bookkeeping and is dropped
        explicitly rather than ignored -- see `convert`.
        """
        self.put(f"codec.rvq.{index}.embed", self.take(self.dialect.codebook(index)))


def read_audiocraft_spec(cfg):
    seanet = audiocraft_ckpt.require(cfg, "seanet")
    encodec = audiocraft_ckpt.require(cfg, "encodec")

    autoencoder = encodec.get("autoencoder", "seanet")
    if autoencoder != "seanet":
        raise ConversionError(f"unsupported autoencoder {autoencoder!r} (expected 'seanet')")

    quantizer = encodec.get("quantizer", "no_quant")
    if quantizer != "no_quant":
        # The two codecs that exist arrive one format each: MelodyFlow's is audiocraft's
        # own and quantizer-free, MusicGen's is HuggingFace's and RVQ. An audiocraft-format
        # RVQ codec would be a third combination nothing has ever produced, so rather than
        # carry untested guesses about its tensor names, fail and say so.
        raise ConversionError(
            f"quantizer {quantizer!r} in an audiocraft checkpoint is not supported. "
            "MusicGen's RVQ codec is loaded through HuggingFace transformers -- convert "
            "the downloaded facebook/encodec_32khz model instead.")

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
    if int(encodec["channels"]) != spec["channels"]:
        raise ConversionError("encodec.channels disagrees with seanet.channels")
    return finish_spec(spec)


def read_hf_spec(config):
    """`transformers.EncodecConfig`, mapped onto the same spec.

    HF names everything differently but means the same thing, with one exception worth
    calling out: `upsampling_ratios` is in *decoder* order and the encoder iterates it
    reversed -- exactly like audiocraft's `ratios` -- so it maps across unchanged.
    """
    def want(key, value):
        actual = config.get(key)
        if actual != value:
            raise ConversionError(f"unsupported {key}={actual!r} (this converter handles "
                                  f"{value!r})")

    want("model_type", "encodec")
    want("norm_type", "weight_norm")
    want("pad_mode", "reflect")
    want("use_causal_conv", False)
    want("use_conv_shortcut", False)
    want("normalize", False)          # otherwise encode/decode carry a per-chunk scale
    want("chunk_length_s", None)      # otherwise the input is windowed and overlap-added
    if config.get("trim_right_ratio", 1.0) != 1.0:
        raise ConversionError("trim_right_ratio != 1.0 only means anything for causal convs")

    ratios = [int(r) for r in config["upsampling_ratios"]]
    spec = {
        "quantizer": "rvq",
        "activation": "ELU",          # EncodecResnetBlock hardcodes nn.ELU
        "channels": int(config["audio_channels"]),
        "n_filters": int(config["num_filters"]),
        "n_residual_layers": int(config["num_residual_layers"]),
        "ratios": ratios,
        "kernel_size": int(config["kernel_size"]),
        "residual_kernel_size": int(config["residual_kernel_size"]),
        "last_kernel_size": int(config["last_kernel_size"]),
        "dilation_base": int(config["dilation_growth_rate"]),
        "compress": int(config["compress"]),
        "lstm_layers": int(config["num_lstm_layers"]),
        "latent_dim": int(config["hidden_size"]),
        "sample_rate": int(config["sampling_rate"]),
        "rvq_bins": int(config["codebook_size"]),
    }
    spec["encoder_dim"] = spec["latent_dim"]
    if int(config["codebook_dim"]) != spec["latent_dim"]:
        raise ConversionError(
            f"codebook_dim {config['codebook_dim']} differs from hidden_size "
            f"{spec['latent_dim']}; this converter quantizes the latent directly")

    # How many codebooks the model actually uses is derived, not stored: audiocraft's
    # wrapper takes the largest `target_bandwidths` entry and turns it into a codebook
    # count. For 32 kHz that is 2.2 kbps / (50 Hz * log2(2048)) = 4.
    hop = int(np.prod(ratios))
    frame_rate = spec["sample_rate"] / hop
    bits = np.log2(spec["rvq_bins"])
    counts = [bw * 1000.0 / (frame_rate * bits) for bw in config["target_bandwidths"]]
    if any(abs(c - round(c)) > 1e-3 for c in counts):
        raise ConversionError(f"target_bandwidths {config['target_bandwidths']} do not give "
                              f"whole codebook counts: {counts}")
    spec["rvq_n_q"] = int(round(max(counts)))
    return finish_spec(spec)


def finish_spec(spec):
    """The parts that follow from the rest, and the invariants worth failing on."""
    spec["hop_length"] = int(np.prod(spec["ratios"]))
    if spec["sample_rate"] % spec["hop_length"]:
        raise ConversionError(
            f"sample_rate {spec['sample_rate']} is not a multiple of the hop "
            f"{spec['hop_length']}")
    spec["frame_rate"] = spec["sample_rate"] // spec["hop_length"]
    if spec["quantizer"] == "no_quant" and spec["encoder_dim"] != 2 * spec["latent_dim"]:
        raise ConversionError(
            f"quantizer-free codec should emit mean||scale: encoder dim "
            f"{spec['encoder_dim']} is not 2 x latent dim {spec['latent_dim']}")
    if spec["quantizer"] == "rvq" and spec["encoder_dim"] != spec["latent_dim"]:
        raise ConversionError(
            f"an RVQ codec quantizes the latent directly: encoder dim "
            f"{spec['encoder_dim']} should equal latent dim {spec['latent_dim']}")
    return spec


def load_source(src):
    """Open a codec checkpoint in whichever of the two formats it is, and say which."""
    src = Path(src)
    if src.suffix == ".safetensors" or (src / "model.safetensors").is_file():
        weights = src if src.suffix == ".safetensors" else src / "model.safetensors"
        config_path = weights.parent / "config.json"
        if not config_path.is_file():
            raise ConversionError(f"no config.json beside {weights}")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ConversionError("reading a HuggingFace checkpoint needs safetensors "
                                  "(`pip install safetensors`)") from exc
        config = json.loads(config_path.read_text())
        state = {}
        with safe_open(str(weights), framework="np") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        return state, read_hf_spec(config), HuggingFaceDialect()

    # audiocraft's own format -- but MusicGen's copy of it is a redirect, not a checkpoint.
    try:
        state, cfg = audiocraft_ckpt.load(src)
    except audiocraft_ckpt.CheckpointError as exc:
        pointer = _pretrained_pointer(src)
        if pointer:
            raise ConversionError(
                f"{src.name} is a pointer to {pointer!r}, not a checkpoint: MusicGen's "
                f"codec is loaded through HuggingFace transformers. Convert the downloaded "
                f"model instead, e.g. --src "
                f"~/.cache/huggingface/hub/models--{pointer.replace('/', '--')}"
                f"/snapshots/<rev>/model.safetensors") from exc
        raise
    return state, read_audiocraft_spec(cfg), AudiocraftDialect()


def _pretrained_pointer(path):
    """The `pretrained` field of an audiocraft codec redirect, if that is what this is."""
    try:
        import torch
        with torch.serialization.safe_globals(audiocraft_ckpt._safe_globals(torch)):
            pkg = torch.load(str(path), map_location="cpu", weights_only=True)
        return pkg.get("pretrained") if isinstance(pkg, dict) else None
    except Exception:
        return None


def convert(src, out, weight_type="f32"):
    if weight_type != "f32":
        # The codec is ~57M parameters; there is little to gain by shrinking it, and the
        # conv stack is the part of the pipeline most sensitive to precision.
        raise ConversionError("only f32 is supported for the codec")
    state, spec, dialect = load_source(src)

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
    if spec["quantizer"] == "rvq":
        w.add_uint32("ac.codec.rvq_n_q", spec["rvq_n_q"])
        w.add_uint32("ac.codec.rvq_bins", spec["rvq_bins"])

    e = Emitter(state, w, dialect)
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
    e.conv("codec.encoder.in", dialect.encoder.format(idx))
    for stage in range(len(spec["ratios"])):
        for j in range(n_res):
            idx += 1
            residual(f"codec.encoder.stage.{stage}.res.{j}", dialect.encoder.format(idx))
        idx += 1
        act(f"codec.encoder.stage.{stage}.snake", dialect.encoder.format(idx))
        idx += 1
        e.conv(f"codec.encoder.stage.{stage}.down", dialect.encoder.format(idx))
    if lstm_layers:
        idx += 1
        e.lstm("codec.encoder.lstm", dialect.encoder.format(idx), lstm_layers)
    idx += 1
    act("codec.encoder.out_snake", dialect.encoder.format(idx))
    idx += 1
    e.conv("codec.encoder.out", dialect.encoder.format(idx))

    # ---- decoder: conv, LSTM, then (act, upsample, residuals) per ratio, act, conv.
    idx = 0
    e.conv("codec.decoder.in", dialect.decoder.format(idx))
    if lstm_layers:
        idx += 1
        e.lstm("codec.decoder.lstm", dialect.decoder.format(idx), lstm_layers)
    for stage in range(len(spec["ratios"])):
        idx += 1
        act(f"codec.decoder.stage.{stage}.snake", dialect.decoder.format(idx))
        idx += 1
        e.conv_transpose(f"codec.decoder.stage.{stage}.up", dialect.decoder.format(idx))
        for j in range(n_res):
            idx += 1
            residual(f"codec.decoder.stage.{stage}.res.{j}", dialect.decoder.format(idx))
    idx += 1
    act("codec.decoder.out_snake", dialect.decoder.format(idx))
    idx += 1
    e.conv("codec.decoder.out", dialect.decoder.format(idx))

    # ---- residual vector quantizer: one codebook per level, applied to the residual.
    if spec["quantizer"] == "rvq":
        for i in range(spec["rvq_n_q"]):
            e.codebook(i)
        # Everything else under `quantizer.` is EMA bookkeeping from training --
        # `cluster_size`, `embed_avg`, `inited`. Dropped by name rather than by silence, so
        # a checkpoint carrying something genuinely new still trips the check below.
        for key in list(state):
            head = key.split(".")[0]
            if head in ("quantizer", "quantiser") and key not in e.consumed:
                tail = key.rsplit(".", 1)[-1]
                if tail in ("cluster_size", "embed_avg", "inited"):
                    e.consumed.add(key)

    unexpected = sorted(set(state) - e.consumed)
    if unexpected:
        raise ConversionError(
            f"{len(unexpected)} unrecognized codec tensor(s): " + ", ".join(unexpected[:8]))

    w.add_uint64("ac.codec.parameter_count", e.params)
    rvq = spec["quantizer"] == "rvq"
    basename = "encodec" if rvq else "melodyflow-vae"
    name = "EnCodec 32 kHz" if rvq else "MelodyFlow SEANet VAE"
    gguf_meta.add_general(w, basename=basename, name=name, n_params=e.params,
                          license_id="mit" if rvq else "cc-by-nc-4.0")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"output": str(out), "tensor_count": e.count, "parameter_count": e.params,
            "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="audiocraft compression_state_dict.bin, or a HuggingFace "
                         "model.safetensors / snapshot directory (MusicGen's EnCodec)")
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
