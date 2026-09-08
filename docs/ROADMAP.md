# Roadmap

The goal is to replace the two remaining PyTorch services in
[gary4local](https://github.com/betweentwomidnights/gary-localhost-installer) — **gary**
(MusicGen `generate_continuation`, port 8000) and **terry** (MelodyFlow `edit`, port 8002)
— with native C++, so the shipped product needs no Python at all. `sa3.cpp` already covers
the other three model families.

Scope is deliberately narrow: **only what those two services actually call.** No melody or
style conditioning, no stereo codebook pattern, no `extend_stride` window (gary always
requests exactly 30 s), no `top_p` (gary always passes 0.0).

MelodyFlow goes first. It maps onto the shape `sa3.cpp` already proves — T5 → DiT →
autoencoder, full-sequence graphs, no autoregression — so it lands terry complete while
de-risking the shared SEANet/LSTM work before MusicGen's incremental-decode problem.

| phase | scope | milestone | state |
|---|---|---|---|
| 0 | repo, ggml pin, shared headers, T5-base encoder + tokenizer | `ac-textenc` matches torch at cossim ≥ 0.9999 | **done** — F32 1.000000000, F16 0.999999934 ([docs/T5.md](T5.md)) |
| 1 | `ac/seanet.h`, `ac/lstm.h`, MelodyFlow VAE (encode + decode) | VAE round-trip matches torch on 30 s stereo | **done** — encode and decode both cossim 1.0000000 ([docs/MELODYFLOW_VAE.md](MELODYFLOW_VAE.md)) |
| 2 | MelodyFlow DiT (RoPE, `add_zero_attn`, additive timestep, U-ViT skips) | one velocity prediction at cossim ≥ 0.9999 | **done** — F32 cossim 1.0000000 end to end ([docs/MELODYFLOW_DIT.md](MELODYFLOW_DIT.md)) |
| 3 | sway schedule, euler/midpoint, CFG, regularized inversion | `mf-edit` reproduces terry's euler/25/0.12/2/1/0.2 | **done** — audio cossim 0.9999179 at 30 s ([docs/MELODYFLOW_EDIT.md](MELODYFLOW_EDIT.md)) |
| 4 | MusicGen LM + KV cache + delay pattern + CFG + top-k | greedy 30 s generation matches token-for-token | next |
| 5 | EnCodec 32 kHz encode + decode | `mg-generate --continue` reproduces `generate_continuation` | |
| 6 | `terry-server` :8002, `gary-server` :8000, quantized tiers, GGUF publication | drop-in for the Python services in gary4local | |

Deferred on purpose: MusicGen LoRA training. `sa3.cpp`'s trainer is already generic
(functional LoRA, gradient checkpointing, quantized-base `out_prod`), so a MusicGen target
is mostly a new loss and dataset adapter — but it waits until inference parity lands. Note
gary4local has no MusicGen LoRA today; customisation is full `thepatch/*` finetunes, so a
converter for the raw audiocraft `state_dict.bin` + `xp.cfg` format matters more.

## Two things that are net-new relative to sa3.cpp

**A KV cache.** `sa3.cpp` has none — every model there is a diffusion or flow model with
full-sequence graphs. MusicGen needs ~1500 sequential steps (×2 for CFG) over a graph built
once and re-executed. This is the largest single piece of engineering in the project.

**An LSTM.** Both codecs put a 2-layer LSTM in the SEANet bottleneck and ggml has no LSTM
op. Resolved in Phase 1: the host-side implementation was written and measured first, spent
2.4x longer on the bottleneck than on the whole convolutional stack around it, and was
replaced by unrolling the recurrence into the graph — 2.6-2.7x faster end to end, and one
graph per direction on any backend. See [docs/MELODYFLOW_VAE.md](MELODYFLOW_VAE.md).

**A third, found in Phase 1:** `ggml_conv_1d` builds its im2col matrix in F16, which rounds
*activations*, not just weights. Over sixteen stacked SEANet convolutions that cost 30x our
parity gate. `nn::conv_1d_f32` is the same computation with an F32 im2col; the memory cost
is a large transient buffer, which will want tiling for long inputs on small GPUs.

**A fourth, found in Phase 2:** MelodyFlow's rotary frequency buffer has been round-tripped
through bfloat16 in the checkpoint, and torch rotates with those rounded values. Recomputing
the closed form does not match. The converter keeps the stored table and feeds it to
`ggml_rope_ext` as `freq_factors` with `freq_base = 1`.

**A fifth, and the only one needing a fork change:** ggml creates its cuBLAS handle with
`CUBLAS_TF32_TENSOR_OP_MATH`, so every F32 GEMM on CUDA computes at ten mantissa bits, with
no way to opt out from the calling side. That put the VAE encoder below the parity gate on
GPU. `feature/audiocraft-cuda-tf32-v0.17.0` adds a default-preserving `GGML_CUDA_TF32=0`
opt-out; `sa3.cpp` and `acestep.cpp` get rebuilt and confirmed unchanged against it before
the branch is published. See [GGML_FORK.md](GGML_FORK.md).

**A sixth, found in Phase 3:** `ggml_gallocr` frees an input tensor's block as soon as its
last consumer has run — `ggml_gallocr_free_node` exempts only `GGML_TENSOR_FLAG_OUTPUT`. A
graph built once and executed many times therefore gets exactly one execution's use out of
an input written at construction; after that a later node's scratch may be on top of it. The
first forward is right and every one after it is quietly wrong. `DitRunner::predict`
re-uploads every input before every execution and `tests/graph_reuse_test.cpp` pins the
contract. Phase 4's KV cache depends on the same pattern.

## Parity details that are easy to get wrong

These produce plausible-but-wrong audio rather than an error, so they are called out here
and asserted in tests where possible:

- MelodyFlow's `add_zero_attn: true` prepends one all-zero key/value in **both** self- and
  cross-attention, in every layer. It is not an optional attention sink: the unconditional
  CFG branch masks every real key to `-inf`, so without it the softmax is NaN.
- MelodyFlow's U-ViT skips pick their projection with `idx % n_skip`, which is not aligned
  with the LIFO pairing and reuses two projections. Trained that way; do not tidy it.
- The feed-forward GELU is the erf form (`F.gelu`, `approximate='none'`), not ggml's
  default tanh approximation.
- MelodyFlow's editing pass uses **`cfg_coef` 4.0**, the `FlowModel` constructor default —
  `get_dit_model` builds from `cfg.transformer_lm`, which carries no `cfg_coef`, so the
  checkpoint's `classifier_free_guidance.inference_coef: 3.0` is never read. The inversion
  pass forces `cfg_coef = 0` because `target_flowstep < source_flowstep`.
- MusicGen passes **no cross-attention mask**; see [docs/T5.md](T5.md).
- MusicGen's timestep-delayed codebook pattern means the AR loop starts at
  `pattern.get_first_step_with_timesteps(start_offset)`, so the first model call is a
  whole-prefix prefill.
