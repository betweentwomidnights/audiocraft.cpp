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

### Where the null branch's zeros go

That second fact has a consequence that cost an afternoon. The unconditional branch's
cross-attention source is all zeros — but **where** the zeroing happens matters.
`output_proj` — which is `lm.cond_proj` here — *does* have a bias, and audiocraft zeroes
the context *after* applying it. Feeding zeros *into* `lm.cond_proj` instead leaves its bias
behind and is a different model, one that still generates plausible audio.

So `mg/pipeline.cpp` projects the real context once and leaves the null stream's slice of
the result at zero, which is exactly what torch does.

There is a tempting shortcut here that guidance batching takes away. With a genuinely
all-zero context and no biases on the attention projections, every key and value in the null
branch's cross-attention is zero, the softmax is uniform over zeros, and the block
contributes nothing to the residual — so the whole thing can be skipped, saving 24 of the
model's 72 attention blocks per guided step. That was worth doing when the streams were two
separate forwards. Batched they share one graph, so the blocks get built either way; the
zeros just flow through them. Batching wins by more than the skip did.

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

## An empty description is not a short one

gary sends `descriptions=None` whenever the caller typed nothing. The service layer defaults
the field to `""`, `get_model_description` treats that as falsy, and only the two
`gary_orchestra` models have a description of their own — so for the other twelve, an
unprompted request generates **unguided**. None of our parity runs took that path until late:
every reference comparison used a real prompt.

That is worth knowing at the product level too, because unguided is audibly worse. A listening
check on vanya put it bluntly: with a description the takes read as the model you expect, and
without one they wander badly. If gary4juce is sending a prompt on most requests, that is why
the difference has not been obvious.

`T5Conditioner` treats an absent description as `""`, collects it in `empty_idx`, zeroes its
attention mask, and multiplies the *projected* embeddings by that mask:

```python
entries   = [xi if xi is not None else "" for xi in x]
empty_idx = [i for i, xi in enumerate(entries) if xi == ""]
mask[empty_idx, :] = 0
embeds = output_proj(t5(...)) * mask.unsqueeze(-1)
```

So the conditional context is **all zeros** — the same thing the null branch carries — and
`uncond + (cond - uncond) * coef` collapses to `uncond`. Torch computes guidance and throws
it away; unprompted MusicGen is unguided MusicGen.

Running T5 over `""` and guiding toward the result instead is a different model. It is also
the obvious implementation, and it is what we shipped through Phase 5: greedy decoding with
no description **diverged from torch at the first generated token** (1375/2400), while the
same run with a description was exact. `mg_use_guidance` in `mg/pipeline.h` is the fix, and
`tests/conditioner_test.cpp` pins both halves of the rule.

Two things to not get clever about. The emptiness test is exact equality with `""`:
`normalize_text` is false in both checkpoints, so `" "` is a description the model was
trained to attend to. And dropping guidance here is not an optimisation with a quality
cost — it is the arithmetic, and it happens to make the unprompted path considerably
faster, since one stream with no cross-attention replaces two with it. Measured back to back
on one card: **1.7x at F32, 2.0x at F16**, and 1.8x on CPU.

## Guidance batching

Classifier-free guidance needs two predictions per step, one conditioned on the text and one
on nothing. They share their input tokens and their weights and differ only in what
cross-attention reads, so they are one forward of batch 2 — which is what audiocraft does,
and what the cache's `n_seq` axis is for.

What keeps it a small change rather than a rewrite is that the two streams differ *only* in
the cross-attention context. `lm.cond_proj` is applied once at setup into a
`[dim, n_ctx, n_seq]` tensor whose second stream is left at zero; the token embeddings and
positions broadcast across the batch (`ggml_add(pos_emb, x)`, positions the wider operand);
every other tensor in the graph just grows a fourth dimension. Sampling then reads the two
logit blocks out of one output and combines them on the host.

The win is not the 2x audiocraft advertises. The arithmetic is the same either way — the
matmuls get twice as wide instead of running twice — so what is saved is per-call overhead
and half the graph builds, which is worth more the smaller the model and the shorter the
step. Measured below: **1.43x on CUDA, 1.30x on CPU.**

Greedy decoding makes this checkable exactly. The change touches attention layout in every
layer, and the tokens have to stay identical or it is wrong: CPU F32 and CUDA F32 both still
produce **6000/6000** against the torch reference.

## The CUDA bug that greedy parity could not see

For a while the unguided path on CUDA produced audio with no rhythm in it: token entropy 8.38
against the CPU's 7.2, and — the giveaway — the *same* to within 0.04 across every seed,
where torch varies by half a bit. A continuation of a drum loop came back with no beat at all.

The cause was in the CUDA backend, not here. `ggml_cuda_cpy` picks a tiled transpose kernel
when `can_be_transposed`, and that kernel writes `dst[imat*n + row*ne00 + col]` — contiguous,
ignoring `nb10..nb13`. The condition only inspected the *source*, so a transposing copy into
a strided view was dispatched to it and landed in the wrong addresses. Our V cache write is
exactly that copy, and the cache was being corrupted on every decode step.

Three coincidences kept it hidden, and each one is worth remembering:

- **`src0->ne[3] == 1` is part of the condition**, so it fired only with a guidance batch of
  one. Every parity run we had used a text prompt, which means `n_seq = 2`, which was exact.
- **A single token makes `nb01 == element_size`**, so the prefill (301 tokens at once) took
  the correct path and only the decode steps were wrong. First-step logits matched torch to
  7e-05 while the rest of the generation was garbage.
- **Greedy decoding never exercises the sampler**, and sampled decoding never gets a
  reference to compare against. The failure lived in the gap between the two.

`tests/kv_attention_test.cpp` pins it: one decode step's attention, run on every hostable
backend, required to agree with the CPU. It is the shape the whole autoregressive loop runs
in — a one-column query against a strided history — and it is not a shape any
full-sequence forward reaches. The fix is a one-line addition of `ggml_is_contiguous(src1)`
in the ggml fork.

The lesson for the parity methodology: **token-exact agreement on one configuration says
nothing about the others.** Guided was exact end to end while unguided was broken from the
second token, on the same binary, the same weights and the same prompt.

## Speed

30 s of audio: 1202 decode steps, 1203 forwards. RTX 5070 Laptop 8 GB, Core Ultra 9 275HX.
Every row below was measured in **one sitting on an otherwise quiet machine**, torch included,
because that turned out to matter (see the note under the table).

| | decode | steps/s | vs torch |
|---|---|---|---|
| torch CUDA, fp16 + xformers, with a description | 27.5 s | 44 | 1.0x |
| torch CUDA, fp16 + xformers, unprompted | 27.4 s | 44 | 1.0x |
| audiocraft.cpp CUDA F32, with a description | 18.7 s | 64.3 | 1.47x |
| audiocraft.cpp CUDA F16, with a description | **16.4 s** | **73.3** | **1.68x** |
| audiocraft.cpp CUDA F32, unprompted | 9.9 s | 121.6 | 2.78x |
| audiocraft.cpp CUDA F16, unprompted | **8.0 s** | **151.0** | **3.45x** |
| audiocraft.cpp CPU F32, with a description | 64.5 s | 18.6 | 0.43x |
| audiocraft.cpp CPU F32, unprompted | 36.9 s | 32.5 | 0.74x |

**torch takes the same time either way.** An empty description makes guidance a no-op, but
audiocraft still builds the batch of two and computes it (see above), so an unprompted
request costs it a full guided forward for nothing. Skipping that is most of the gap in the
last two CUDA rows, and it is gary's default for twelve of the fourteen `thepatch` models.

**We are faster than torch here** — unlike MelodyFlow, where we are 2x behind. The difference
is what each side is good at: MelodyFlow is 750-token full-sequence forwards where xformers'
fused attention dominates, while MusicGen is single-token steps where per-call overhead
dominates and ggml's is lower. torch's number includes the codec decode (well under a second
of it); ours excludes model loading, which is another 1-3 s.

An earlier revision of this table claimed 2.43x by comparing our 13.5 s against a 32.7 s torch
figure timed on a different day. Both numbers were real; the ratio was not. On a laptop, with
a DAW and a webcam utility holding graphics contexts and the card idling 17 C warmer, the same
binaries came in about 25% slower across the board while every ratio held. **Time both sides
in one sitting or do not quote the multiple.**

The remaining wins are an **F16 KV cache** — 591 MB is most of what an 8 GB card has to
spare, and halving it also halves the bandwidth attention reads every step — and quantized
weights, which nothing has tried yet.

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
- **The service is `musicgen-server`**; see [docs/SERVICES.md](SERVICES.md).
- **Model coverage.** Only `thepatch/vanya_ai_dnb_0.1` (small) has been converted and run.
  The converter is driven entirely by `xp.cfg` and rejects anything it does not recognise, so
  medium and large should convert unchanged — but "should" is not "did".
- **Speed.** An F16 KV cache and quantized tiers, per above.

## Deferred

- **`extend_stride`.** Generation beyond `max_duration` slides a window and re-primes from
  the tail. gary always requests exactly 30 s, so the branch is dead code for us.
- **`top_p`, melody and style conditioning, `cfg_coef_beta`, `two_step_cfg`.** None are
  reachable from gary's parameters.
- **Seed parity with torch.** `src/rng.h` is reproducible within audiocraft.cpp and shares
  nothing with torch's stream. Matching it means reimplementing Philox and `torch.multinomial`
  and is out of scope; greedy decoding is what the parity check uses instead.
- **MusicGen LoRA training.** Still waiting on inference, as planned.
