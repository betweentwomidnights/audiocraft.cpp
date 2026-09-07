# MelodyFlow's flow-matching DiT

`facebook/melodyflow-t24-30secs` is a 962M-parameter `FlowModel`
(audiocraft `models/flow.py`): a pre-norm transformer over 128-channel audio latents,
dim 1536, 24 layers, 24 heads of 64, feed-forward 6144, with

- **RoPE self-attention** using the frequencies stored in the checkpoint,
- **cross-attention** onto T5-base states projected 768 → 1536,
- **U-ViT skip connections** across the stack, 11 of them,
- a **timestep embedding that is added** to the normed input before each attention block —
  there is no adaLN or modulation anywhere in this model,
- no bias on any projection; LayerNorm (eps 1e-5) carries weight and bias.

Input and output are rescaled around the transformer: `x = z / sqrt(t² + (1-t)²)` going in,
`× sqrt(2)` coming out, so the DiT's own activations stay unit-scaled at every flow step.

## Conversion

```bash
MF=~/.cache/huggingface/hub/models--facebook--melodyflow-t24-30secs/snapshots/*/
.venv/Scripts/python.exe tools/convert_melodyflow_dit.py \
  --src "$MF/state_dict.bin" \
  --out models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --out-type f16
```

334 tensors, 962,425,632 parameters. `--out-type f32` gives the reference build (3.9 GB);
F16 is 1.9 GB and is what fits alongside activations on an 8 GB GPU.

Norms, biases, the rotary table and the latent statistics stay F32 in both builds — they
are a rounding error's worth of file size and they are where half precision actually shows.

The converter refuses anything it was not written for: a non-`layer_norm` norm, post-norm,
sinusoidal position, a missing cross-attention, absent skips, `add_zero_attn` off, a
non-GELU activation, causal attention, `layer_scale`, biased projections, qk layer norm, a
fuser doing anything but a single `description` cross condition, or a conditioner that is
not `t5-base`. It also fails on any unconsumed source tensor.

## Three things that are silently load-bearing

### The rotary frequencies are bfloat16-rounded, and torch uses them that way

`RotaryEmbedding` registers `1 / max_period ** (arange(0, dim, 2) / dim)` as a buffer. In
this checkpoint that buffer has been round-tripped through bfloat16 at some point — the
weights have not — so the stored values are exactly `bfloat16(analytic)`: index 1 is
`0.75`, not `0.74989421`.

`load_state_dict` loads those values, and torch rotates with them. Recomputing the closed
form in C++ would therefore *not* match. The relative error reaches 3.7e-3, and the angle
at a position is `pos × freq`, so over MelodyFlow's 750-frame window the second frequency
alone drifts about 0.08 radians by the end of the sequence.

So the converter keeps them. ggml computes
`theta_i = (pos × freq_base^(-2i/n_dims)) / freq_factors[i]`, so passing `freq_base = 1`
collapses the first term to 1 for every `i` and the stored frequencies go in verbatim as
`freq_factors[i] = 1 / freq[i]` — no dependence on how ggml accumulates its own power
series. `check_rope_frequencies` still validates the buffer against the closed form at a
1% tolerance, which accepts the half-precision rounding that is really in the file while
rejecting a genuinely different rotary setup.

Pairs are interleaved — `view_as_complex(x.reshape(..., -1, 2))` — which is
`GGML_ROPE_TYPE_NORMAL`, not NeoX.

### `add_zero_attn` is what makes the null branch defined

Every attention block prepends one all-zero key and value:
`F.pad(k, (0, 0, 1, 0))` on a `[batch, heads, time, dim]` tensor.

It is easy to read as an optional attention sink and drop. It is not optional. In the
unconditional branch of classifier-free guidance, `ClassifierFreeGuidanceDropout` replaces
the description with `None`, which tokenizes to `""`, whose attention mask is all zeros —
and MelodyFlow passes `log(mask)` as the cross-attention mask, so **every real key is
`-inf`**. Without the zero key the softmax has no finite entry and the null branch is NaN.
With it, the only surviving key has value zero, so cross-attention contributes exactly
nothing — which is the intended meaning of "unconditional".

`ac-dit --uncond` exercises this path, and it matches the reference exactly.

### The feed-forward GELU is the erf form

`activation: gelu` reaches `nn.TransformerEncoderLayer` as the string `"gelu"`, which torch
maps to `F.gelu` with `approximate='none'`. `ggml_gelu` is the tanh approximation; this
uses `ggml_gelu_erf`.

## The skip wiring looks wrong and has to stay that way

```python
if skip_connections and idx > len(layers) / 2:
    x = torch.cat([x, states.pop()], dim=-1)
    x = self.skip_projections[idx % len(self.skip_projections)](x)
x = layer(x)
if skip_connections and idx < len(layers) / 2 - 1:
    states.append(x)
```

For 24 layers that pushes the outputs of layers 0–10 and pops them at layers 13–23, LIFO:
layer 13 concatenates layer 10's output, layer 23 concatenates layer 0's. But the
projection index is `idx % 11`, which over the consuming layers runs
`2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1` — not aligned with the pairing, and reusing projections
0 and 1. It reads like an off-by-something in the original. The weights were trained
through it, so it is the contract. `tests/dit_skip_test.cpp` transcribes the reference loop
and pins ours against it, including that exact projection sequence, precisely so nobody
tidies it later.

## Measured parity

Reference: `tools/dump_mf_dit_refs.py`, which loads the real `FlowModel` through
`load_dit_model_melodyflow` and calls `FlowModel.forward` — the same call the ODE solver
makes at every step — conditioning through the model's own `condition_provider`. The latent
is a fixed-seed standard normal, which is the distribution the DiT sees at every flow step.

| case | build | cossim | max abs err |
|---|---|---:|---:|
| 250 frames, t=0.37, supplied T5 states | F32 | 1.0000000 | 1.1e-05 |
| 250 frames, t=0.37, supplied T5 states | F16 | 0.9999991 | 7.8e-03 |
| 250 frames, t=0.37, **our own T5** end to end | F32 | 1.0000000 | 9.8e-06 |
| 250 frames, t=0.37, null branch (`--uncond`) | F32 | 1.0000000 | 6.9e-06 |
| 750 frames (30 s), t=0.12 | F32 | 1.0000000 | 1.7e-05 |
| 750 frames (30 s), t=0.12 | F16 | 0.9999992 | 6.8e-03 |

F32 reaches float32 round-off, so the graph is exact; the F16 residual is entirely weight
quantization accumulated over 24 layers. The end-to-end row runs our own tokenizer, T5
encoder and `cond_proj` rather than being handed the reference's hidden states, so the
whole conditioning path is covered.

## Speed

One forward at the full 750-frame window, this machine:

| backend | build | forward | vs torch |
|---|---|---:|---:|
| CPU (Core Ultra 9 275HX) | F32 | 7.29 s | 1.0000000 |
| CPU | F16 | 7.67 s | 0.9999992 |
| CUDA (RTX 5070 Laptop, 8 GB) | F32 | 0.298 s | 0.9999975 |
| CUDA | F16 | 0.234 s | 0.9999768 |

F16 is marginally *slower* on CPU — dequantisation is not free and these matmuls were not
bandwidth-bound — which matches what `sa3.cpp` documents about quantisation on CPU. On GPU
the two are the same speed at this size, so F16 buys VRAM headroom rather than throughput.

**CUDA is 32x faster and measurably less accurate**, because ggml runs every F32 GEMM at
TF32 — see [GGML_FORK.md](GGML_FORK.md#open-every-f32-gemm-on-cuda-silently-runs-at-tf32).
Both rows still clear the 0.9999 gate for a single forward. With TF32 disabled by the local
patch documented there, F32 goes to 0.9999975 (at +34% time) and F16 to 0.9999768 (free) —
so if this needs tightening later, F16 is where it is cheapest.

terry's default edit is 25 euler steps with `regularize_iters=2`, and the two-pass `edit()`
comes to 125 forwards (see `flow_forward_count`). That is roughly 15 minutes on this CPU
and 28 seconds on this GPU, which is what makes the GPU path the real one.

## Reproducing

```bash
# reference (needs torch and audiocraft -- use terry's venv)
MFDIR=/c/dev/gary-localhost-installer/services/melodyflow
PYTHONPATH="C:\\dev\\gary-localhost-installer\\services\\melodyflow" \
  $MFDIR/env/Scripts/python.exe tools/dump_mf_dit_refs.py \
    --out refdit --frames 250 --t 0.37

# ours, against the reference's T5 states
build/bin/Release/ac-dit --dit models/melodyflow-dit-t24-1.0B-v1.0-F32.gguf \
  --cond refdit/mf_dit_cond.f32 --latent refdit/mf_dit_z.f32 --t 0.37 \
  --out cppdit/mf_dit_velocity.f32

# ours, end to end from the prompt
build/bin/Release/ac-dit --dit models/melodyflow-dit-t24-1.0B-v1.0-F32.gguf \
  --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf \
  --prompt "Lively accordion music with a European folk feeling" \
  --latent refdit/mf_dit_z.f32 --t 0.37 --out cppdit/mf_dit_velocity.f32

.venv/Scripts/python.exe tools/cossim.py --ref refdit --cpp cppdit
```

Add `--uncond` to both sides for the null branch.
