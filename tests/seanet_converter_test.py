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


def snake(state, prefix, channels):
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
    state = {}

    # --- encoder: conv, then (residuals, act, downsample) per reversed ratio, LSTM, act, conv
    idx, mult = 0, 1
    wn_conv(state, f"encoder.model.{idx}", n_filters, s["channels"], s["kernel_size"])
    for ratio in reversed(ratios):
        dim = mult * n_filters
        for _ in range(n_res):
            idx += 1
            p = f"encoder.model.{idx}"
            snake(state, f"{p}.block.0", dim)
            wn_conv(state, f"{p}.block.1", dim // compress, dim, s["residual_kernel_size"])
            snake(state, f"{p}.block.2", dim // compress)
            wn_conv(state, f"{p}.block.3", dim, dim // compress, 1)
        idx += 1
        snake(state, f"encoder.model.{idx}", dim)
        idx += 1
        wn_conv(state, f"encoder.model.{idx}", dim * 2, dim, ratio * 2)
        mult *= 2
    bottleneck = mult * n_filters
    if s["lstm"]:
        idx += 1
        lstm(state, f"encoder.model.{idx}", bottleneck, s["lstm"])
    idx += 1
    snake(state, f"encoder.model.{idx}", bottleneck)
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
        snake(state, f"decoder.model.{idx}", dim)
        idx += 1
        wn_convtr(state, f"decoder.model.{idx}", dim, dim // 2, ratio * 2)
        for _ in range(n_res):
            idx += 1
            p = f"decoder.model.{idx}"
            snake(state, f"{p}.block.0", dim // 2)
            wn_conv(state, f"{p}.block.1", dim // 2 // compress, dim // 2,
                    s["residual_kernel_size"])
            snake(state, f"{p}.block.2", dim // 2 // compress)
            wn_conv(state, f"{p}.block.3", dim // 2, dim // 2 // compress, 1)
        mult //= 2
    idx += 1
    snake(state, f"decoder.model.{idx}", n_filters)
    idx += 1
    wn_conv(state, f"decoder.model.{idx}", s["channels"], n_filters, s["last_kernel_size"])
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

    def test_rejects_rvq_for_now(self):
        cfg = config(**{"encodec.quantizer": "rvq"})
        with self.assertRaises(ConversionError):
            self.convert(fixture_state(config()), cfg)

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
