#!/usr/bin/env python3
"""Synthetic coverage for tools/convert_t5.py.

Adapted from sa3.cpp tests/sat_t5_converter_test.py. Runs on a tiny fabricated
checkpoint so it needs no model download and stays a non-GPU CTest.
"""

import gc
import json
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
    from safetensors.numpy import save_file
    from convert_t5 import ConversionError, convert
except ImportError as exc:  # pragma: no cover - environment dependent
    DEPENDENCY_ERROR = str(exc)


def fixture_state(layers=1):
    sd = {
        "shared.weight": np.zeros((8, 4), np.float32),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
            np.zeros((4, 2), np.float32),
        "encoder.final_layer_norm.weight": np.ones(4, np.float32),
    }
    for i in range(layers):
        s = f"encoder.block.{i}."
        sd[s + "layer.0.layer_norm.weight"] = np.ones(4, np.float32)
        sd[s + "layer.1.layer_norm.weight"] = np.ones(4, np.float32)
        sd[s + "layer.1.DenseReluDense.wi.weight"] = np.zeros((6, 4), np.float32)
        sd[s + "layer.1.DenseReluDense.wo.weight"] = np.zeros((4, 6), np.float32)
        for name in ("q", "k", "v", "o"):
            sd[s + f"layer.0.SelfAttention.{name}.weight"] = np.zeros((4, 4), np.float32)
    return sd


def config(**overrides):
    cfg = {"d_model": 4, "num_layers": 1, "num_heads": 2, "d_kv": 2, "d_ff": 6,
           "vocab_size": 8, "relative_attention_num_buckets": 4,
           "relative_attention_max_distance": 128, "layer_norm_epsilon": 1e-6,
           "pad_token_id": 0, "eos_token_id": 1,
           "feed_forward_proj": "relu", "is_gated_act": False}
    cfg.update(overrides)
    return cfg


def tokenizer():
    return {"model": {"unk_id": 2,
                      "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0],
                                ["▁", -1.0], ["a", -2.0], ["▁a", -0.5]]},
            "added_tokens": [{"id": 0, "content": "<pad>"}, {"id": 1, "content": "</s>"}]}


@unittest.skipIf(DEPENDENCY_ERROR is not None,
                 f"converter dependencies unavailable: {DEPENDENCY_ERROR}")
class T5ConverterTest(unittest.TestCase):
    def run_fixture(self, state, cfg=None, max_length=256):
        """Convert a fixture and return (result, kv dict, tensor-name set).

        GGUFReader memory-maps the output, and on Windows an open mapping blocks the
        temporary directory from being removed -- so we read everything we need out of
        the reader, drop it, and let the TemporaryDirectory context close cleanly.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "m.safetensors"
            cfg_path = root / "config.json"
            tok_path = root / "tokenizer.json"
            out = root / "m.gguf"
            save_file(state, str(src))
            cfg_path.write_text(json.dumps(cfg or config()), encoding="utf-8")
            tok_path.write_text(json.dumps(tokenizer()), encoding="utf-8")
            result = convert(src, cfg_path, tok_path, out, max_length=max_length)
            reader = GGUFReader(str(out))
            kv = {key: field.contents() for key, field in reader.fields.items()}
            names = {t.name for t in reader.tensors}
            del reader
            gc.collect()
        return result, kv, names

    def test_converts_encoder_and_tokenizer(self):
        result, kv, names = self.run_fixture(fixture_state())
        # embed + relative bias + final norm + 8 per layer
        self.assertEqual(result["tensor_count"], 3 + 8)
        self.assertEqual(result["token_count"], 6)
        self.assertIn("te.embed.weight", names)
        self.assertIn("te.relative_attention_bias.weight", names)
        self.assertIn("te.norm.weight", names)
        for suffix in ("attn_norm.weight", "q.weight", "k.weight", "v.weight",
                       "o.weight", "ffn_norm.weight", "wi.weight", "wo.weight"):
            self.assertIn(f"te.0.{suffix}", names)
        self.assertEqual(kv["ac.architecture"], "audiocraft")
        self.assertEqual(kv["ac.format_version"], 1)
        self.assertEqual(kv["ac.t5.dim"], 4)
        self.assertEqual(kv["ac.t5.layers"], 1)
        self.assertEqual(kv["ac.t5.relative_max_distance"], 128)
        self.assertEqual(kv["ac.t5.max_length"], 256)
        self.assertEqual(kv["tok.model"], "sentencepiece-unigram")
        self.assertEqual(kv["tok.pad_id"], 0)
        self.assertEqual(kv["tok.eos_id"], 1)
        self.assertEqual(kv["tok.unk_id"], 2)

    def test_multi_layer_tensor_count(self):
        result, _, _ = self.run_fixture(fixture_state(layers=3), cfg=config(num_layers=3))
        self.assertEqual(result["tensor_count"], 3 + 3 * 8)

    def test_rejects_gated_feed_forward(self):
        # google/t5-v1_1-base and flan-t5 are gated-GELU; audiocraft uses plain t5-base.
        with self.assertRaises(ConversionError):
            self.run_fixture(fixture_state(),
                             cfg=config(feed_forward_proj="gated-gelu", is_gated_act=True))

    def test_rejects_missing_tensor(self):
        state = fixture_state()
        del state["encoder.block.0.layer.1.DenseReluDense.wo.weight"]
        with self.assertRaises(ConversionError):
            self.run_fixture(state)

    def test_rejects_unconsumed_encoder_tensor(self):
        state = fixture_state()
        state["encoder.block.0.layer.0.SelfAttention.surprise.weight"] = np.zeros((4, 4), np.float32)
        with self.assertRaises(ConversionError):
            self.run_fixture(state)

    def test_rejects_non_unigram_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "m.safetensors"
            cfg_path = root / "config.json"
            tok_path = root / "tokenizer.json"
            save_file(fixture_state(), str(src))
            cfg_path.write_text(json.dumps(config()), encoding="utf-8")
            tok_path.write_text(json.dumps({"model": {"type": "BPE"}}), encoding="utf-8")
            with self.assertRaises(ConversionError):
                convert(src, cfg_path, tok_path, root / "m.gguf")

    def test_rejects_bad_max_length(self):
        with self.assertRaises(ConversionError):
            self.run_fixture(fixture_state(), max_length=0)


if __name__ == "__main__":
    unittest.main()
