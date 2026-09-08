#!/usr/bin/env python3
"""Synthetic coverage for tools/convert_seanet.py.

The valuable part to test is the module-index walk. audiocraft's SEANet is an
`nn.Sequential`, so every tensor name is a bare integer index into a list whose layout
depends on `n_residual_layers`, the number of ratios and whether there is an LSTM. Get the
walk wrong and the converter happily maps stage 2's weights onto stage 1 -- the shapes even
agree in places -- and the codec produces plausible, wrong audio.

So the fixtures below build a state dict the way audiocraft's constructor does, from the
same config, and the test asserts the converter consumed every key and emitted the names
`src/ac/seanet.h` asks for. Nothing here needs torch or a checkpoint on disk.
"""

import gc
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DEPENDENCY_ERROR = None
try:
    import numpy as np
    from gguf import GGUFReader
    import audiocraft_ckpt
    import convert_seanet
    from convert_seanet import ConversionError
except ImportError as exc:  # pragma: no cover - environment dependent
    DEPENDENCY_ERROR = str(exc)


def config(**overrides):
    """A miniature MelodyFlow codec: same shape, tiny widths."""
    seanet = {
        "dimension": 4, "channels": 2, "causal": False, "n_filters": 2,
        "n_residual_layers": 1, "ratios": [4, 2], "activation": "snake",
        "activation_params": {"alpha": 1.0}, "norm": "weight_norm", "norm_params": {},
        "kernel_size": 7, "residual_kernel_size": 3, "last_kernel_size": 7,
        "dilation_base": 2, "pad_mode": "reflect", "true_skip": True, "compress": 2,
        "lstm": 2, "disable_norm_outer_blocks": 0,
        "decoder": {"trim_right_ratio": 1.0, "final_activation": None},
        "encoder": {"dimension": 8},
    }
    encodec = {"autoencoder": "seanet", "quantizer": "no_quant", "sample_rate": 16,
               "channels": 2, "causal": False, "renormalize": False}
    cfg = {"seanet": seanet, "encodec": encodec}
    for path, value in overrides.items():
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def wn_conv(state, prefix, out_ch, in_ch, kernel):
    """A weight-normed StreamableConv1d, as torch stores it after parametrization."""
    rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
    state[f"{prefix}.conv.conv.bias"] = rng.standard_normal(out_ch, dtype=np.float32)
    state[f"{prefix}.conv.conv.parametrizations.weight.original0"] = (
        rng.standard_normal((out_ch, 1, 1), dtype=np.float32) + 2.0)
    state[f"{prefix}.conv.conv.parametrizations.weight.original1"] = (
        rng.standard_normal((out_ch, in_ch, kernel), dtype=np.float32) + 1.0)


def wn_convtr(state, prefix, in_ch, out_ch, kernel):
    rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
    state[f"{prefix}.convtr.convtr.bias"] = rng.standard_normal(out_ch, dtype=np.float32)
    state[f"{prefix}.convtr.convtr.parametrizations.weight.original0"] = (
        rng.standard_normal((in_ch, 1, 1), dtype=np.float32) + 2.0)
    state[f"{prefix}.convtr.convtr.parametrizations.weight.original1"] = (
        rng.standard_normal((in_ch, out_ch, kernel), dtype=np.float32) + 1.0)


def snake(state, prefix, channels, activation="snake"):
    """Snake owns a per-channel alpha; ELU owns nothing. The fixture has to match, or the
    converter would be handed keys it has no reason to consume."""
    if activation != "snake":
        return
    state[f"{prefix}.alpha"] = np.ones((1, channels, 1), np.float32)


def lstm(state, prefix, hidden, layers):
    for layer in range(layers):
        state[f"{prefix}.lstm.weight_ih_l{layer}"] = np.zeros((4 * hidden, hidden), np.float32)
        state[f"{prefix}.lstm.weight_hh_l{layer}"] = np.zeros((4 * hidden, hidden), np.float32)
        state[f"{prefix}.lstm.bias_ih_l{layer}"] = np.zeros(4 * hidden, np.float32)
        state[f"{prefix}.lstm.bias_hh_l{layer}"] = np.zeros(4 * hidden, np.float32)


def fixture_state(cfg):
    """Build the state dict audiocraft's SEANetEncoder/Decoder constructors produce."""
    s = cfg["seanet"]
    n_filters, ratios = s["n_filters"], s["ratios"]
    n_res, compress = s["n_residual_layers"], s["compress"]
    act = s["activation"]
    state = {}

    # --- encoder: conv, then (residuals, act, downsample) per reversed ratio, LSTM, act, conv
    idx, mult = 0, 1
    wn_conv(state, f"encoder.model.{idx}", n_filters, s["channels"], s["kernel_size"])
    for ratio in reversed(ratios):
        dim = mult * n_filters
        for _ in range(n_res):
            idx += 1
            p = f"encoder.model.{idx}"
            snake(state, f"{p}.block.0", dim, act)
            wn_conv(state, f"{p}.block.1", dim // compress, dim, s["residual_kernel_size"])
            snake(state, f"{p}.block.2", dim // compress, act)
            wn_conv(state, f"{p}.block.3", dim, dim // compress, 1)
        idx += 1
        snake(state, f"encoder.model.{idx}", dim, act)
        idx += 1
        wn_conv(state, f"encoder.model.{idx}", dim * 2, dim, ratio * 2)
        mult *= 2
    bottleneck = mult * n_filters
    if s["lstm"]:
        idx += 1
        lstm(state, f"encoder.model.{idx}", bottleneck, s["lstm"])
    idx += 1
    snake(state, f"encoder.model.{idx}", bottleneck, act)
    idx += 1
    wn_conv(state, f"encoder.model.{idx}", s["encoder"]["dimension"], bottleneck,
            s["last_kernel_size"])

    # --- decoder: conv, LSTM, then (act, upsample, residuals) per ratio, act, conv
    idx, mult = 0, 2 ** len(ratios)
    wn_conv(state, f"decoder.model.{idx}", mult * n_filters, s["dimension"], s["kernel_size"])
    if s["lstm"]:
        idx += 1
        lstm(state, f"decoder.model.{idx}", mult * n_filters, s["lstm"])
    for ratio in ratios:
        dim = mult * n_filters
        idx += 1
        snake(state, f"decoder.model.{idx}", dim, act)
        idx += 1
        wn_convtr(state, f"decoder.model.{idx}", dim, dim // 2, ratio * 2)
        for _ in range(n_res):
            idx += 1
            p = f"decoder.model.{idx}"
            snake(state, f"{p}.block.0", dim // 2, act)
            wn_conv(state, f"{p}.block.1", dim // 2 // compress, dim // 2,
                    s["residual_kernel_size"])
            snake(state, f"{p}.block.2", dim // 2 // compress, act)
            wn_conv(state, f"{p}.block.3", dim // 2, dim // 2 // compress, 1)
        mult //= 2
    idx += 1
    snake(state, f"decoder.model.{idx}", n_filters, act)
    idx += 1
    wn_conv(state, f"decoder.model.{idx}", s["channels"], n_filters, s["last_kernel_size"])
    return state


# --- the HuggingFace dialect -----------------------------------------------------------
#
# MusicGen's codec arrives as `transformers.EncodecModel`, which is a faithful port of the
# same SEANet under different names. The fixtures below are the same miniature topology
# spelled HF's way, which is what makes the cross-dialect test below meaningful: one schema,
# two readers, and if either drifts the emitted names stop matching.


def hf_config(**overrides):
    """A miniature EnCodec, matching `config()`'s topology exactly."""
    cfg = {
        "model_type": "encodec", "audio_channels": 2, "num_filters": 2,
        "num_residual_layers": 1, "upsampling_ratios": [4, 2], "kernel_size": 7,
        "residual_kernel_size": 3, "last_kernel_size": 7, "dilation_growth_rate": 2,
        "compress": 2, "num_lstm_layers": 2, "hidden_size": 4, "codebook_dim": 4,
        "codebook_size": 8, "sampling_rate": 16, "norm_type": "weight_norm",
        "pad_mode": "reflect", "use_causal_conv": False, "use_conv_shortcut": False,
        "normalize": False, "chunk_length_s": None, "trim_right_ratio": 1.0,
        # 8 codes is 3 bits per codebook; at 2 Hz two codebooks is 12 bits/s = 0.012 kbps.
        "target_bandwidths": [0.012],
    }
    cfg.update(overrides)
    return cfg


def hf_conv(state, prefix, out_ch, in_ch, kernel):
    """HF stores weight norm in the legacy `weight_g`/`weight_v` form."""
    rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
    state[f"{prefix}.conv.bias"] = rng.standard_normal(out_ch, dtype=np.float32)
    state[f"{prefix}.conv.weight_g"] = rng.standard_normal((out_ch, 1, 1), np.float32) + 2.0
    state[f"{prefix}.conv.weight_v"] = (
        rng.standard_normal((out_ch, in_ch, kernel), dtype=np.float32) + 1.0)


def hf_convtr(state, prefix, in_ch, out_ch, kernel):
    """ConvTranspose1d: the weight is [in, out, k] and weight_g is per *input* channel."""
    rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
    state[f"{prefix}.conv.bias"] = rng.standard_normal(out_ch, dtype=np.float32)
    state[f"{prefix}.conv.weight_g"] = rng.standard_normal((in_ch, 1, 1), np.float32) + 2.0
    state[f"{prefix}.conv.weight_v"] = (
        rng.standard_normal((in_ch, out_ch, kernel), dtype=np.float32) + 1.0)


def hf_lstm(state, prefix, hidden, layers):
    for layer in range(layers):
        state[f"{prefix}.lstm.weight_ih_l{layer}"] = np.zeros((4 * hidden, hidden), np.float32)
        state[f"{prefix}.lstm.weight_hh_l{layer}"] = np.zeros((4 * hidden, hidden), np.float32)
        state[f"{prefix}.lstm.bias_ih_l{layer}"] = np.zeros(4 * hidden, np.float32)
        state[f"{prefix}.lstm.bias_hh_l{layer}"] = np.zeros(4 * hidden, np.float32)


def hf_fixture_state(config, n_q=2):
    """The state dict `transformers.EncodecModel` produces for `hf_config()`."""
    n_filters = config["num_filters"]
    ratios = config["upsampling_ratios"]
    n_res, compress = config["num_residual_layers"], config["compress"]
    state = {}

    idx, mult = 0, 1
    hf_conv(state, f"encoder.layers.{idx}", n_filters, config["audio_channels"],
            config["kernel_size"])
    for ratio in reversed(ratios):
        dim = mult * n_filters
        for _ in range(n_res):
            idx += 1
            p = f"encoder.layers.{idx}"
            hf_conv(state, f"{p}.block.1", dim // compress, dim,
                    config["residual_kernel_size"])
            hf_conv(state, f"{p}.block.3", dim, dim // compress, 1)
        idx += 2                                    # the ELU owns a slot but no parameters
        hf_conv(state, f"encoder.layers.{idx}", dim * 2, dim, ratio * 2)
        mult *= 2
    bottleneck = mult * n_filters
    if config["num_lstm_layers"]:
        idx += 1
        hf_lstm(state, f"encoder.layers.{idx}", bottleneck, config["num_lstm_layers"])
    idx += 2
    hf_conv(state, f"encoder.layers.{idx}", config["hidden_size"], bottleneck,
            config["last_kernel_size"])

    idx, mult = 0, 2 ** len(ratios)
    hf_conv(state, f"decoder.layers.{idx}", mult * n_filters, config["hidden_size"],
            config["kernel_size"])
    if config["num_lstm_layers"]:
        idx += 1
        hf_lstm(state, f"decoder.layers.{idx}", mult * n_filters, config["num_lstm_layers"])
    for ratio in ratios:
        dim = mult * n_filters
        idx += 2
        hf_convtr(state, f"decoder.layers.{idx}", dim, dim // 2, ratio * 2)
        for _ in range(n_res):
            idx += 1
            p = f"decoder.layers.{idx}"
            hf_conv(state, f"{p}.block.1", dim // 2 // compress, dim // 2,
                    config["residual_kernel_size"])
            hf_conv(state, f"{p}.block.3", dim // 2, dim // 2 // compress, 1)
        mult //= 2
    idx += 2
    hf_conv(state, f"decoder.layers.{idx}", config["audio_channels"], n_filters,
            config["last_kernel_size"])

    for i in range(n_q):
        p = f"quantizer.layers.{i}.codebook"
        rng = np.random.default_rng(1000 + i)
        state[f"{p}.embed"] = rng.standard_normal(
            (config["codebook_size"], config["codebook_dim"]), dtype=np.float32)
        # EMA bookkeeping from training. The converter must drop these by name.
        state[f"{p}.embed_avg"] = state[f"{p}.embed"].copy()
        state[f"{p}.cluster_size"] = np.ones(config["codebook_size"], np.float32)
        state[f"{p}.inited"] = np.ones(1, np.float32)
    return state


@unittest.skipIf(DEPENDENCY_ERROR is not None,
                 f"converter dependencies unavailable: {DEPENDENCY_ERROR}")
class SeanetConverterTest(unittest.TestCase):
    def convert(self, state, cfg):
        """Run the converter against an in-memory checkpoint.

        GGUFReader memory-maps its file and on Windows that blocks the temporary directory
        from being removed, so everything needed is read out before the reader is dropped.
        """
        original = audiocraft_ckpt.load
        audiocraft_ckpt.load = lambda src: (state, cfg)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "codec.gguf"
                result = convert_seanet.convert("ignored", out)
                reader = GGUFReader(str(out))
                kv = {key: field.contents() for key, field in reader.fields.items()}
                names = {t.name for t in reader.tensors}
                shapes = {t.name: tuple(int(v) for v in t.shape if v) for t in reader.tensors}
                data = {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}
                del reader
                gc.collect()
            return result, kv, names, shapes, data
        finally:
            audiocraft_ckpt.load = original

    def test_walks_the_module_indices(self):
        cfg = config()
        result, kv, names, shapes, _ = self.convert(fixture_state(cfg), cfg)

        self.assertEqual(kv["ac.architecture"], "audiocraft")
        self.assertEqual(kv["ac.codec.quantizer"], "no_quant")
        self.assertEqual(kv["ac.codec.activation"], "snake")
        self.assertEqual(list(kv["ac.codec.ratios"]), [4, 2])
        self.assertEqual(kv["ac.codec.hop_length"], 8)
        self.assertEqual(kv["ac.codec.frame_rate"], 2)     # sample_rate 16 / hop 8
        self.assertEqual(kv["ac.codec.latent_dim"], 4)
        self.assertEqual(kv["ac.codec.encoder_dim"], 8)
        self.assertEqual(kv["ac.codec.lstm_layers"], 2)

        for name in ("codec.encoder.in.weight", "codec.encoder.out.weight",
                     "codec.decoder.in.weight", "codec.decoder.out.weight",
                     "codec.encoder.out_snake.alpha", "codec.decoder.out_snake.alpha_recip",
                     "codec.encoder.lstm.1.w_hh", "codec.decoder.lstm.0.bias"):
            self.assertIn(name, names)
        for stage in (0, 1):
            self.assertIn(f"codec.encoder.stage.{stage}.down.weight", names)
            self.assertIn(f"codec.decoder.stage.{stage}.up.weight_col2im", names)
            for part in ("snake1.alpha", "conv1.weight", "snake2.alpha_recip", "conv2.bias"):
                self.assertIn(f"codec.encoder.stage.{stage}.res.0.{part}", names)
                self.assertIn(f"codec.decoder.stage.{stage}.res.0.{part}", names)

        # The encoder walks the ratios reversed, so stage 0 is the ratio-2 downsample
        # (kernel 4) and stage 1 the ratio-4 one (kernel 8). This is the assertion that
        # actually catches an off-by-one in the walk: a wrong index would still find a
        # tensor, just the neighbouring one, with a different kernel.
        # ggml order is [kernel, in, out]; GGUFReader reports the same.
        self.assertEqual(shapes["codec.encoder.stage.0.down.weight"][0], 4)
        self.assertEqual(shapes["codec.encoder.stage.1.down.weight"][0], 8)
        # The decoder walks them forward, so its stage 0 is the ratio-4 upsample.
        self.assertEqual(shapes["codec.decoder.stage.0.up.bias"][0], 4)   # 8 channels -> 4
        self.assertEqual(shapes["codec.decoder.stage.1.up.bias"][0], 2)   # 4 channels -> 2

        # Every source tensor must be consumed; convert() raises otherwise, so reaching
        # here already proves it, but pin the count so an accidental drop is visible.
        self.assertEqual(result["tensor_count"], len(names))

    def test_alpha_reciprocal_is_precomputed(self):
        # Snake is x + (alpha + 1e-9).reciprocal() * sin(alpha * x)**2; the graph reads the
        # reciprocal rather than dividing per call, so it has to be right here.
        cfg = config()
        state = fixture_state(cfg)
        alphas = np.array([2.5, 0.5, 4.0, 1.0, 0.25, 8.0, 0.125, 16.0], np.float32)
        # The encoder's final snake: index 0 conv, two stages of (res, snake, down),
        # then the LSTM -- so 1 + 2*3 + 1 = 8.
        state["encoder.model.8.alpha"] = alphas.reshape(1, -1, 1)
        _, _, names, _, data = self.convert(state, cfg)
        self.assertIn("codec.encoder.out_snake.alpha_recip", names)
        np.testing.assert_allclose(data["codec.encoder.out_snake.alpha"], alphas, rtol=1e-6)
        np.testing.assert_allclose(data["codec.encoder.out_snake.alpha_recip"],
                                   1.0 / (alphas.astype(np.float64) + 1e-9), rtol=1e-6)

    def test_handles_more_residual_layers(self):
        cfg = config(**{"seanet.n_residual_layers": 3})
        _, _, names, _, _ = self.convert(fixture_state(cfg), cfg)
        for j in range(3):
            self.assertIn(f"codec.encoder.stage.0.res.{j}.conv1.weight", names)

    def test_handles_no_lstm(self):
        cfg = config(**{"seanet.lstm": 0})
        _, kv, names, _, _ = self.convert(fixture_state(cfg), cfg)
        self.assertEqual(kv["ac.codec.lstm_layers"], 0)
        self.assertFalse(any(n.startswith("codec.encoder.lstm") for n in names))

    def convert_hf(self, state, config):
        """Run the converter against an in-memory HuggingFace checkpoint."""
        original = convert_seanet.load_source
        convert_seanet.load_source = lambda src: (
            state, convert_seanet.read_hf_spec(config), convert_seanet.HuggingFaceDialect())
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "codec.gguf"
                result = convert_seanet.convert("ignored", out)
                reader = GGUFReader(str(out))
                kv = {key: field.contents() for key, field in reader.fields.items()}
                names = {t.name for t in reader.tensors}
                shapes = {t.name: tuple(int(v) for v in t.shape if v) for t in reader.tensors}
                data = {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}
                del reader
                gc.collect()
            return result, kv, names, shapes, data
        finally:
            convert_seanet.load_source = original

    def test_hf_walks_the_module_indices(self):
        config = hf_config()
        result, kv, names, _, _ = self.convert_hf(hf_fixture_state(config), config)
        self.assertEqual(kv["ac.codec.quantizer"], "rvq")
        self.assertEqual(kv["ac.codec.activation"], "ELU")
        self.assertEqual(kv["ac.codec.frame_rate"], 2)
        self.assertEqual(kv["ac.codec.hop_length"], 8)
        # 0.012 kbps / (2 Hz * log2(8)) = 2 codebooks.
        self.assertEqual(kv["ac.codec.rvq_n_q"], 2)
        self.assertEqual(kv["ac.codec.rvq_bins"], 8)
        for i in range(2):
            self.assertIn(f"codec.rvq.{i}.embed", names)
        # ELU owns no parameters, so nothing snake-shaped may be emitted for it.
        self.assertFalse(any(n.endswith(".alpha") for n in names))
        self.assertEqual(result["spec"]["ratios"], [4, 2])

    def test_hf_drops_the_quantizer_ema_state(self):
        """`cluster_size`, `embed_avg` and `inited` are training bookkeeping.

        They have to be dropped *by name*: leaving them unconsumed would trip the
        converter's "unrecognized tensor" check, and consuming everything under
        `quantizer.` by prefix would hide a genuinely new field.
        """
        config = hf_config()
        state = hf_fixture_state(config)
        self.assertIn("quantizer.layers.0.codebook.embed_avg", state)
        _, _, names, _, _ = self.convert_hf(state, config)
        self.assertFalse(any("embed_avg" in n or "cluster_size" in n or "inited" in n
                             for n in names))

        state["quantizer.layers.0.codebook.something_new"] = np.zeros(4, np.float32)
        with self.assertRaises(ConversionError):
            self.convert_hf(state, config)

    def test_both_dialects_emit_the_same_schema(self):
        """The whole point of one converter: two source formats, one set of tensor names.

        `src/ac/seanet.h` reads both codecs through the same names, so if either reader
        drifts the other model breaks silently. Held here by converting the same miniature
        topology from both formats and comparing what came out.
        """
        ac_cfg = config(**{"seanet.activation": "ELU"})
        _, _, ac_names, ac_shapes, _ = self.convert(fixture_state(ac_cfg), ac_cfg)
        hf_cfg = hf_config()
        _, _, hf_names, hf_shapes, _ = self.convert_hf(hf_fixture_state(hf_cfg), hf_cfg)

        rvq_only = {n for n in hf_names if n.startswith("codec.rvq.")}
        self.assertEqual(hf_names - rvq_only, ac_names)

        # One shape legitimately differs, and it is the bottleneck: MelodyFlow's codec has
        # no quantizer, so its encoder emits mean||scale at twice the latent width. Pinned
        # rather than excused, so a change anywhere else still fails.
        widened = {"codec.encoder.out.weight", "codec.encoder.out.bias"}
        for name in sorted(ac_names):
            if name in widened:
                self.assertNotEqual(ac_shapes[name], hf_shapes[name], name)
            else:
                self.assertEqual(ac_shapes[name], hf_shapes[name], name)
        self.assertEqual(ac_shapes["codec.encoder.out.bias"][0],
                         2 * hf_shapes["codec.encoder.out.bias"][0])

    def test_rejects_unsupported_hf_topology(self):
        for override in ({"normalize": True},          # adds a per-chunk scale we do not carry
                         {"chunk_length_s": 1.0},      # windows and overlap-adds the input
                         {"use_causal_conv": True},
                         {"use_conv_shortcut": True},
                         {"norm_type": "time_group_norm"},
                         {"pad_mode": "constant"},
                         {"trim_right_ratio": 0.5},
                         {"codebook_dim": 8}):         # must equal hidden_size
            cfg = hf_config(**override)
            with self.subTest(override=override), self.assertRaises(ConversionError):
                self.convert_hf(hf_fixture_state(hf_config()), cfg)

    def test_rejects_fractional_codebook_counts(self):
        """`target_bandwidths` has to divide into whole codebooks; audiocraft asserts it too."""
        cfg = hf_config(target_bandwidths=[0.031])
        with self.assertRaises(ConversionError):
            self.convert_hf(hf_fixture_state(hf_config()), cfg)


    def test_rejects_unsupported_topology(self):
        for override in ({"seanet.causal": True},
                         {"seanet.true_skip": False},
                         {"seanet.norm": "none"},
                         {"seanet.pad_mode": "constant"},
                         {"seanet.activation": "ReLU"},
                         {"seanet.disable_norm_outer_blocks": 1},
                         {"encodec.autoencoder": "oobleck"}):
            cfg = config(**override)
            with self.subTest(override=override), self.assertRaises(ConversionError):
                self.convert(fixture_state(config()), cfg)

    def test_rejects_decoder_final_activation(self):
        cfg = config()
        cfg["seanet"]["decoder"]["final_activation"] = "Tanh"
        with self.assertRaises(ConversionError):
            self.convert(fixture_state(config()), cfg)

    def test_rejects_a_latent_that_is_not_mean_and_scale(self):
        # A quantizer-free codec must emit twice the latent width.
        cfg = config(**{"seanet.encoder": {"dimension": 6}})
        with self.assertRaises(ConversionError):
            self.convert(fixture_state(config()), cfg)

    def test_rejects_unconsumed_tensors(self):
        cfg = config()
        state = fixture_state(cfg)
        state["encoder.model.99.conv.conv.bias"] = np.zeros(4, np.float32)
        with self.assertRaises(ConversionError):
            self.convert(state, cfg)

    def test_rejects_missing_tensor(self):
        cfg = config()
        state = fixture_state(cfg)
        del state["decoder.model.0.conv.conv.bias"]
        with self.assertRaises(ConversionError):
            self.convert(state, cfg)

    def test_fuse_weight_norm_matches_the_definition(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((3, 4, 5), dtype=np.float32)
        g = rng.standard_normal((3, 1, 1), dtype=np.float32) + 2.0
        fused = convert_seanet.fuse_weight_norm(g, v)
        norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True))
        want = g.astype(np.float64) * v.astype(np.float64) / norm
        np.testing.assert_allclose(fused, want, rtol=1e-6, atol=1e-6)
        # Each output channel of the fused weight has norm |g|, which is the property
        # weight_norm exists to give.
        np.testing.assert_allclose(np.linalg.norm(fused.reshape(3, -1), axis=1),
                                   np.abs(g).ravel(), rtol=1e-5)

    def test_fuse_weight_norm_rejects_a_zero_slice(self):
        v = np.zeros((2, 3, 3), np.float32)
        g = np.ones((2, 1, 1), np.float32)
        with self.assertRaises(ConversionError):
            convert_seanet.fuse_weight_norm(g, v)

    def test_col2im_layout(self):
        # torch [in, out, kernel] -> numpy [out*kernel, in], so ggml sees [in, kernel*out]
        # and ggml_col2im_1d can read whole output-channel blocks.
        w = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        out = convert_seanet.conv_transpose_for_col2im(w)
        self.assertEqual(out.shape, (12, 2))
        np.testing.assert_array_equal(out[:, 0], w[0].reshape(-1))
        np.testing.assert_array_equal(out[:, 1], w[1].reshape(-1))


if __name__ == "__main__":
    unittest.main()
