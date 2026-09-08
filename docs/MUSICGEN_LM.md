# MusicGen's language model

gary's whole job is `generate_continuation`: take the last few seconds of a track, encode
them, and let a 420M-parameter transformer keep going for 30 s. The transformer is the hard
part and it is what this phase covers. Turning its output back into audio is EnCodec's job
and lands in Phase 5.

```
text ──► T5-base ──► cond_proj ─┐
                                ▼
 prompt codes ─► delay pattern ─► prefill (301 tokens) ─► 1202 single-token steps ─► codes
                                   \                                              /
                                    ` KV cache, two streams (conditional + null) ´
```

`src/mg/pattern.h` is the interleaving, `src/mg/kv_cache.h` the history, `src/mg/lm.h` the
model, `src/mg/sampling.h` the token choice, `src/mg/pipeline.cpp` the loop.

## What gary actually asks for

From `g4laudio.py`'s `AudioConfig`, unchanged since we started:

| | |
|---|---|
| `duration` | **30.0 s**, always |
| `prompt_duration` | 6.0 s (the first 6 s for `process_audio`, the last 6 s for `continue_music`) |
| `top_k` / `top_p` | 250 / **0.0** |
| `temperature` | 1.0 |
| `cfg_coef` | 3.0 |
| `two_step_cfg` | False |
| `remove_prompts` | False — the output includes the prompt |

Two of those shrink the work considerably. `duration` never exceeds `max_duration`, so the
`extend_stride` sliding-window branch in `genmodel.py` is **never taken**; and `top_p` is
always 0, so top-k is the only sampling path that needs to exist.

`cfg_coef` here is unambiguous, unlike MelodyFlow's: `set_generation_params` passes it
explicitly at generation time, and 3.0 is the checkpoint's `inference_coef`, audiocraft's
default and gary's value all at once.

## The delay pattern

EnCodec emits `n_q = 4` codebooks per timestep. The LM does not model them in parallel — it
interleaves them so codebook `q` at sequence step `s` carries timestep `s - 1 - q`:

```
codes                    sequence (S = special token 2048)
[[1, 2, 3, 4],           [[S, 1, 2, 3, 4, S, S, S],
 [1, 2, 3, 4],    ->      [S, S, 1, 2, 3, 4, S, S],
 [1, 2, 3, 4],            [S, S, S, 1, 2, 3, 4, S],
 [1, 2, 3, 4]]            [S, S, S, S, 1, 2, 3, 4]]
```

The point is causality *within* a step: when the model predicts codebook 2 at step `s`, its
context already contains codebook 1 for the same timestep. 1500 timesteps become 1504
sequence steps, and a 6 s prompt (300 timesteps) means the loop starts at step 301 — so the
first model call is a 301-token prefill and the remaining 1202 are single tokens.

audiocraft builds this by materializing a layout of coordinate lists and scatter indexes
over a flattened tensor. `mg/pattern.h` is the closed form, which is exactly the kind of
thing that is easy to get subtly wrong — get a delay off by one and the model still produces
audio, just misaligned across codebooks. `tests/pattern_test.cpp` checks it against fixtures
from the real `DelayedPatternProvider`, driven through the same `build_pattern_sequence` /
`revert_pattern_sequence` calls `generate` makes.

Two positions in the sequence carry no timestep and must read back as the special token: the
model never emits 2048, so without forcing them the tail of the sequence fills with real
codes that revert to nothing.

## The KV cache

Re-attending the whole prefix for 1202 steps would be quadratic, so every layer's keys and
values live in a backend buffer of their own — deliberately not the graph arena, which
`ggml_gallocr` recycles between executions (see [MELODYFLOW_EDIT.md](MELODYFLOW_EDIT.md)).

**K and V are stored in different layouts on purpose.**

```
K  [head_dim, capacity, n_head]   what mul_mat(k, q) wants
V  [capacity, head_dim, n_head]   already transposed
```

`nn::sdpa` has to `ggml_cont(ggml_permute(v))` before the second matmul. With a cache that
transpose runs over the *entire history* on every decode step: at a full 30 s context that
is 148 MB of copy per step across 24 layers, for nothing. `nn::sdpa_vt` consumes the
transposed layout directly and the copy disappears. The trade-off is that
`ggml_flash_attn_ext` wants V the other way round, so a flash path for a cached model would
have to transpose back — nothing needs that yet.

At the shipped shape the cache is **296 MB per stream**, 591 MB for both. F16 would halve it
and is untried.

**The graph is rebuilt for every forward.** It has to be: a cache view's offset and extent
encode how much history there is, and ggml bakes those in at build time. Building costs
about 0.3 ms for this stack against 16 ms of compute, so it is ~2% overhead. The alternative
— attending over the full capacity with a mask and keeping one static graph — doubles the
attention work to save that. The context memory and the allocator are reused, so only the
graph objects are rebuilt, not the buffers behind them.

## Two things that are silently load-bearing

**The sinusoidal position divides by `half_dim - 1`.** `create_sin_embedding` computes
`pos / max_period ** (i / (half_dim - 1))`, which is off by one from the usual definition
*and* off by one from the timestep embedding in the same codebase (`mf/dit.h` divides by
`half`). The GGUF carries `ac.lm.pos_denominator` so the reader cannot quietly guess the
conventional form.

**Cross-attention gets no mask.** `T5Conditioner.forward` zeroes padded positions after
`output_proj` and `ConditionFuser` passes no mask, so T5 padding is attended over as zero
vectors rather than being excluded. This is the opposite of MelodyFlow, which masks with a
real −inf.

### The null branch skips cross-attention entirely

That second fact has a useful consequence. The unconditional branch's cross-attention source
is all zeros, and the attention projections have no biases, so every key and value is zero,
the softmax is uniform over zeros, and the block contributes exactly nothing to the residual.
So `mg/pipeline.cpp` does not build those 24 blocks at all for the null stream.

**Where the zeroing happens matters, and getting it wrong cost an afternoon.** `output_proj`
— which is `lm.cond_proj` here — *does* have a bias, and audiocraft zeroes the context
*after* applying it. Feeding zeros into `lm.cond_proj` instead leaves its bias behind and is
a different model. The first version of the `--uncond-cross` cross-check did exactly that,
disagreed with the skip, and briefly looked like the skip was the bug — when in fact the skip
matched torch and the check did not.

`mg-generate --uncond-cross` now builds the blocks and feeds them a *projected* zero. Result:
bit-identical to skipping, and both match torch. The optimization is worth about 24 of the
model's 72 attention blocks per guided step.

## Measured parity

`thepatch/vanya_ai_dnb_0.1` (musicgen-small shape: dim 1024, 24 layers, 16 heads, 420M
parameters), greedy decoding, guidance 3.0, against **torch on CPU**.

| | timesteps | tokens matching |
|---|---|---|
| text-to-music, "drum and bass" | 100 | **100.00%** (400 / 400) |
| continuation, 100-timestep prompt | 200 | **100.00%** (800 / 800) |
| continuation, 300-timestep prompt (gary's 6 s) | **1500** | **100.00%** (6000 / 6000) |

Token-for-token over the full 30 s, prompt included. That is the strongest form this check
takes: the delay pattern, the KV cache, the guidance combination, the sinusoidal positions
and the argmax all have to be right simultaneously, and any one of them being wrong diverges
within a few steps.

### Why CUDA does not match token for token

It does not, and it should not be expected to:

| | first prediction's logits | tokens matching |
|---|---|---|
| CPU F32 | cossim **1.0000000**, max abs err 8.0e-05 | 100% |
| CUDA F32 | cossim 0.9999992, max abs err 2.8e-02 | diverges at the first generated step |
| CUDA F16 | — | diverges at the first generated step |

The logits are right; `argmax` is discontinuous. Guidance amplifies the gap further —
`uncond + (cond − uncond) · 3` scales a difference by about four — and once one step picks a
different token the sequences are unrelated. A 2.8e-2 absolute error is larger than the top-2
margin often enough that this happens immediately.

This is worth stating plainly because it bounds what "parity" can mean for an autoregressive
model: **arithmetic parity is checkable, output parity is not.** The first prediction's
logits are the right thing to compare — `mg-generate --out-logits` and the dumper's
`mg_logits0.npy` exist for exactly that — and beyond that the question becomes whether the
audio is as good, not whether it is the same.

None of which affects gary, which samples rather than decoding greedily and so produces a
different take on every run regardless.

## Speed

30 s of audio: 1202 decode steps, 2406 forwards (two guidance streams). RTX 5070 Laptop
8 GB, Core Ultra 9 275HX.

| | decode | steps/s | vs torch |
|---|---|---|---|
| torch CUDA, fp16 + xformers (what gary runs) | 32.7 s | 37 | 1.0x |
| audiocraft.cpp CUDA F32 | 23.8 s | 50.4 | **1.37x** |
| audiocraft.cpp CUDA F16 | **19.3 s** | **62.4** | **1.69x** |
| audiocraft.cpp CPU F32 | 82 s | 14.6 | 0.4x |

**We are already faster than torch here** — unlike MelodyFlow, where we are 2x behind. The
difference is what each side is good at: MelodyFlow is 750-token full-sequence forwards where
xformers' fused attention dominates, while MusicGen is single-token steps where per-call
overhead dominates and ggml's is lower. torch's number includes the codec decode (well under
a second of it); ours excludes model loading, which is another 1-3 s.

The obvious remaining win is **batching the two guidance streams into one forward**.
audiocraft does exactly that and calls it "about x2 faster"; here they are two sequential
forwards over two caches. That plus an F16 KV cache is where Phase 6 should start.

## Reproducing

```bash
# convert (needs gguf + torch; the repo venv has both)
.venv/Scripts/python.exe tools/convert_musicgen_lm.py \
  --src ~/.cache/huggingface/hub/models--thepatch--vanya_ai_dnb_0.1/snapshots/*/state_dict.bin \
  --out models/musicgen-vanya-dnb-0.4B-v1.0-F32.gguf --out-type f32 \
  --source-repo thepatch/vanya_ai_dnb_0.1

# reference (gary's venv). --device cpu matters: load_lm_model builds the model in float16
# on CUDA and MusicGen wraps generation in a float16 autocast, so torch-on-GPU is a
# different computation from torch-on-CPU.
PYTHONPATH=.../services/gary .../env/Scripts/python.exe tools/dump_mg_lm_refs.py \
    --out refmg30 --prompt "drum and bass" --input testdata/music.wav \
    --prompt-duration 6.0 --duration 30.0 --device cpu

# ours, from the same prompt codes
mg-generate --lm models/musicgen-vanya-dnb-0.4B-v1.0-F32.gguf \
            --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf \
            --prompt "drum and bass" --prompt-codes refmg30/mg_prompt_codes.i32 \
            --duration 30 --greedy --out-codes cppmg30/mg_codes.i32

python -c "import numpy as np; a=np.fromfile('refmg30/mg_codes.i32',dtype=np.int32); \
           b=np.fromfile('cppmg30/mg_codes.i32',dtype=np.int32); \
           print((a==b).mean())"
```

Shorten with `--duration 2` on both sides while bisecting; the reference at 30 s takes about
seven minutes on CPU.

## Remaining integration gates

- **The prompt codes come from Python.** `mg-generate --prompt-codes` needs a file the
  reference dumper produced, because EnCodec is Phase 5. Until then this is an LM driver,
  not gary's endpoint.
- **`--frame-rate` is a flag with a default of 50.** It belongs to the codec, and Phase 5
  should read it from the EnCodec GGUF rather than trusting an argument.
- **No `gary-server` yet.** Phase 6 owns `:8000`, the job queue and the `session_id` polling.
- **Model coverage.** Only `thepatch/vanya_ai_dnb_0.1` (small) has been converted and run.
  The converter is driven entirely by `xp.cfg` and rejects anything it does not recognise, so
  medium and large should convert unchanged — but "should" is not "did".
- **Speed.** Guidance batching and an F16 cache, per above.

## Deferred

- **`extend_stride`.** Generation beyond `max_duration` slides a window and re-primes from
  the tail. gary always requests exactly 30 s, so the branch is dead code for us.
- **`top_p`, melody and style conditioning, `cfg_coef_beta`, `two_step_cfg`.** None are
  reachable from gary's parameters.
- **Seed parity with torch.** `src/rng.h` is reproducible within audiocraft.cpp and shares
  nothing with torch's stream. Matching it means reimplementing Philox and `torch.multinomial`
  and is out of scope; greedy decoding is what the parity check uses instead.
- **MusicGen LoRA training.** Still waiting on inference, as planned.
