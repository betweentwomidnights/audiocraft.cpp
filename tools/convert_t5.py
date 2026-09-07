#!/usr/bin/env python3
"""Convert the T5-base encoder and its Unigram tokenizer to one GGUF.

MusicGen and MelodyFlow both use audiocraft's `T5Conditioner` with name "t5-base", which
is plain google-t5/t5-base -- ReLU feed-forward, ungated, learned relative position bias.
Adapted from sa3.cpp tools/convert_sat_t5.py; the tensor mapping is identical, only the
metadata namespace differs (sat.* -> ac.*).

One semantic difference from stable-audio-tools: audiocraft does NOT pad prompts to a
fixed length -- `T5Conditioner.tokenize` pads to the longest entry in the batch. So
`ac.t5.max_length` here is a *capacity* bound for graph sizing, not a pad length. Encoding
a prompt at its true token count and encoding it padded-with-mask give identical hidden
states at the valid positions (the encoder is bidirectional and padded keys are -inf
masked), so callers may do either.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from safetensors import safe_open

import gguf_meta

T5_BASE_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"


class ConversionError(ValueError):
    pass


def convert(src, config, tokenizer, out, weight_type="f16", max_length=256):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    tok = json.loads(Path(tokenizer).read_text(encoding="utf-8"))
    if weight_type not in ("f16", "f32"):
        raise ConversionError("weight_type must be f16 or f32")
    wdtype = np.float16 if weight_type == "f16" else np.float32

    if cfg.get("feed_forward_proj", "relu") != "relu" or cfg.get("is_gated_act", False):
        raise ConversionError(
            "this converter targets plain t5-base (ungated ReLU feed-forward); "
            f"got feed_forward_proj={cfg.get('feed_forward_proj')!r}, "
            f"is_gated_act={cfg.get('is_gated_act')!r}")

    dim, layers = int(cfg["d_model"]), int(cfg["num_layers"])
    heads, head_dim = int(cfg["num_heads"]), int(cfg["d_kv"])
    vocab_size = int(cfg["vocab_size"])
    if max_length < 1:
        raise ConversionError("max_length must be positive")

    model = tok.get("model", {})
    vocab = model.get("vocab")
    if not isinstance(vocab, list) or not vocab or not isinstance(vocab[0], list):
        raise ConversionError("tokenizer.json is not a SentencePiece Unigram tokenizer")
    tokens = [str(x[0]) for x in vocab]
    scores = [float(x[1]) for x in vocab]
    for added in tok.get("added_tokens", []):
        idx = int(added["id"])
        if idx < len(tokens):
            tokens[idx] = str(added["content"])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="ac-t5")
    w.add_string("ac.architecture", "audiocraft")
    w.add_uint32("ac.format_version", 1)
    w.add_uint32("ac.t5.dim", dim)
    w.add_uint32("ac.t5.layers", layers)
    w.add_uint32("ac.t5.heads", heads)
    w.add_uint32("ac.t5.head_dim", head_dim)
    w.add_uint32("ac.t5.intermediate", int(cfg["d_ff"]))
    w.add_uint32("ac.t5.vocab", vocab_size)
    w.add_uint32("ac.t5.relative_buckets", int(cfg["relative_attention_num_buckets"]))
    w.add_uint32("ac.t5.relative_max_distance",
                 int(cfg.get("relative_attention_max_distance", 128)))
    w.add_float32("ac.t5.eps", float(cfg["layer_norm_epsilon"]))
    w.add_uint32("ac.t5.max_length", max_length)
    w.add_string("ac.t5.weight_type", weight_type.upper())
    w.add_string("tok.model", "sentencepiece-unigram")
    w.add_array("tok.tokens", tokens)
    w.add_array("tok.scores", scores)
    w.add_uint32("tok.pad_id", int(cfg["pad_token_id"]))
    w.add_uint32("tok.eos_id", int(cfg["eos_token_id"]))
    w.add_uint32("tok.unk_id", int(model.get("unk_id", 2)))
    w.add_bool("tok.add_eos", True)
    w.add_string("tok.metaspace", "▁")

    count = params = 0
    consumed = set()
    with safe_open(str(src), framework="numpy") as f:
        keys = set(f.keys())

        def emit(dst, source, weight=False):
            nonlocal count, params
            if source not in keys:
                raise ConversionError(f"missing T5 tensor: {source}")
            consumed.add(source)
            a = np.ascontiguousarray(
                f.get_tensor(source).astype(wdtype if weight else np.float32))
            w.add_tensor(dst, a)
            count += 1
            params += int(a.size)

        emit("te.embed.weight", "shared.weight", True)
        emit("te.relative_attention_bias.weight",
             "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        for i in range(layers):
            s = f"encoder.block.{i}."
            d = f"te.{i}."
            emit(d + "attn_norm.weight", s + "layer.0.layer_norm.weight")
            for qkvo in ("q", "k", "v", "o"):
                emit(d + qkvo + ".weight", s + f"layer.0.SelfAttention.{qkvo}.weight", True)
            emit(d + "ffn_norm.weight", s + "layer.1.layer_norm.weight")
            emit(d + "wi.weight", s + "layer.1.DenseReluDense.wi.weight", True)
            emit(d + "wo.weight", s + "layer.1.DenseReluDense.wo.weight", True)
        emit("te.norm.weight", "encoder.final_layer_norm.weight")

        encoder_keys = {k for k in keys if k.startswith("encoder.")} | {"shared.weight"}
        unexpected = sorted(encoder_keys - consumed)
        if unexpected:
            raise ConversionError("unrecognized encoder tensors: " + ", ".join(unexpected[:8]))

    w.add_uint64("ac.t5.parameter_count", params)
    gguf_meta.add_general(w, basename="t5-base-encoder", name="T5-base encoder",
                          n_params=params, license_id="apache-2.0")
    gguf_meta.add_source(
        w, "T5-base", "Google", "https://huggingface.co/google-t5/t5-base",
        T5_BASE_REVISION, "model.safetensors")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"output": str(out), "tensor_count": count, "parameter_count": params,
            "token_count": len(tokens)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="t5-base model.safetensors")
    ap.add_argument("--config", required=True, help="t5-base config.json")
    ap.add_argument("--tokenizer", required=True, help="t5-base tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-type", choices=("f16", "f32"), default="f16")
    ap.add_argument("--max-length", type=int, default=256,
                    help="graph-capacity bound in tokens (audiocraft does not pad to a "
                         "fixed length; this only sizes the graph)")
    a = ap.parse_args()
    try:
        r = convert(a.src, a.config, a.tokenizer, a.out, a.out_type, a.max_length)
    except (ConversionError, OSError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(f"error: {exc}")
    print(f"wrote {r['output']} ({r['tensor_count']} tensors, "
          f"{r['parameter_count']:,} parameters, {r['token_count']} tokens, "
          f"{a.out_type.upper()})")


if __name__ == "__main__":
    main()
