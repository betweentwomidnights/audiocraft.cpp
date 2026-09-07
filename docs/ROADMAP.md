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
| 2 | MelodyFlow DiT (RoPE, `add_zero_attn`, additive timestep, U-ViT skips) | one velocity prediction at cossim ≥ 0.9999 | next |
| 3 | sway schedule, euler/midpoint, CFG, regularized inversion | `mf-edit` reproduces terry's euler/25/0.12/2/1/0.2 | |
| 4 | MusicGen LM + KV cache + delay pattern + CFG + top-k | greedy 30 s generation matches token-for-token | |
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

## Parity details that are easy to get wrong

These produce plausible-but-wrong audio rather than an error, so they are called out here
and asserted in tests where possible:

- MelodyFlow's `add_zero_attn: true` prepends one all-zero key/value in **both** self- and
  cross-attention, in every layer.
- MelodyFlow's editing pass uses **`cfg_coef` 4.0**, the `FlowModel` constructor default —
  `get_dit_model` builds from `cfg.transformer_lm`, which carries no `cfg_coef`, so the
  checkpoint's `classifier_free_guidance.inference_coef: 3.0` is never read. The inversion
  pass forces `cfg_coef = 0` because `target_flowstep < source_flowstep`.
- MusicGen passes **no cross-attention mask**; see [docs/T5.md](T5.md).
- MusicGen's timestep-delayed codebook pattern means the AR loop starts at
  `pattern.get_first_step_with_timesteps(start_offset)`, so the first model call is a
  whole-prefix prefill.
