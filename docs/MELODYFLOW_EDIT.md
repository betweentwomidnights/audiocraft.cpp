# MelodyFlow's `edit`

`MelodyFlow.edit` is what terry does to every request, and it is two ODE solves over one
DiT. The source audio's latent is integrated *backwards* to a pivot flow step with no
conditioning and no guidance, then forwards again to 1.0 under the target prompt. The pivot
— `target_flowstep`, 0.05 to 0.2 across terry's presets — is the dial: 0 discards the source
entirely, 1 keeps it.

```
wav ─► SEANet encoder ─► vae_sample ─► normalize ─┐
                                                  ▼
                          inversion  1.0 ──► 0.12   guidance off, KL-regularized, 75 forwards
                                                  │
                          generation 0.12 ─► 1.0    guidance 4.0,   no regularization, 50 forwards
                                                  ▼
                   denormalize ─► SEANet decoder ─► wav
```

`src/mf/solver.h` owns the arithmetic and has no ggml in it; `src/mf/pipeline.{h,cpp}` owns
the graph; `tools/mf-edit.cpp` is the driver. The three models are loaded one at a time —
text encoder, codec, DiT, codec again — because at the 750-frame window each wants gigabytes
of compute buffer and an 8 GB card cannot hold two.

## What terry actually asks for

From `localhost_melodyflow.py::process_audio`, unchanged since we started:

| | euler (default) | midpoint |
|---|---|---|
| `steps` | 25 | 64 |
| `regularize` | true | false |
| `regularize_iters` / `keep_last_k_iters` | 2 / 1 | — |
| `lambda_kl` | 0.2 | — |
| `target_flowstep` | the preset's, or the request's | same |

`cfg_coef` is **4.0**, the `FlowModel` constructor default. `get_dit_model` builds from
`cfg.transformer_lm`, which carries no `cfg_coef`, so the checkpoint's
`classifier_free_guidance.inference_coef: 3.0` is never read. The inversion then forces
guidance to 0 because `target_flowstep < source_flowstep`.

`mf-edit`'s defaults are exactly this row, so

```bash
mf-edit --dit models/melodyflow-dit-t24-1.0B-v1.0-F32.gguf \
        --codec models/melodyflow-vae-48khz-v1.0-F32.gguf \
        --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf \
        --input in.wav --prompt "a dubby reggae bassline" --out out.wav
```

is terry's transform.

## Four things in the loop that look like mistakes

All four are transcribed from `FlowModel.generate` and all four are load-bearing. They are
pinned in `tests/solver_test.cpp`, whose fixtures come from calling audiocraft's *own*
`generate` as an unbound function against a stub velocity field — so the control flow being
checked is theirs, not a transcription of it.

**Regularized steps evaluate the field at the step they are heading for**, `schedule[idx+1]`,
not at the current one. Only when `regularize` is set; the generation pass uses
`current_flowstep` like a normal Euler step.

**The step taken at the end of a regularized step uses a moving average** of the regularized
velocities, weighted `jdx / (k·threshold + sum(range(k)))` — a denominator that happens to
equal the sum of the numerators, so the weights sum to 1. Meanwhile each inner iteration
re-steps from `gen`, not from the previous iterate, using the *latest* velocity rather than
the average. With terry's 2/1 there is exactly one regularized iteration and the average is
a no-op, but the general form is implemented.

**Two configurations audiocraft does not guard, and we reject.** `keep_last_k_iters = 0`
with `regularize` on leaves the moving average at zero, so the solve takes no step at all;
`regularize_iters == keep_last_k_iters == 1` makes the only weight `0/0`. Both throw here
rather than silently producing nothing.

**The midpoint solver's odd steps span two schedule entries** — `schedule[idx+1] -
schedule[idx-1]` — after the even step stashed a half step. Regularization is rejected with
midpoint, matching audiocraft's assert.

## The noise is an input, not a parameter

`edit` draws from the global torch RNG twice: once for the VAE posterior sample of the
encoded input, then twice per regularized solver step — 1 + 2·25 = 51 draws of
[128, 750] at terry's settings. `torch.manual_seed` immediately before the call is what
makes a transform reproducible on the Python side.

That stream cannot be reproduced here, and pretending otherwise would make every comparison
approximate. So `tools/dump_mf_edit_refs.py` generates the noise itself with numpy, writes
it to `mf_edit_noise.f32`, and patches `torch.randn_like` to hand it back; `mf-edit --noise`
replays the same file. Float arithmetic is then the only difference between the two runs.

`mf-edit --dump-noise` records what a normal seeded run consumed, which is how you take a
result you liked and hand it to the reference.

**Seeds do not carry across.** `--seed` drives `src/rng.h`'s mt19937 + Box-Muller, which is
reproducible within audiocraft.cpp on every backend but shares nothing with torch. A seed
from terry will not reproduce terry's output here, and that is not fixable short of
reimplementing torch's Philox.

## The trap that cost an afternoon

A solve runs one graph 125 times. Building and allocating once and re-executing is the
whole point — the DiT graph takes 1 ms to build but its compute buffer is gigabytes — and it
hides this:

> `ggml_gallocr` hands out one arena for the whole graph and frees a value's block as soon as
> its last consumer has run. `ggml_gallocr_free_node` exempts `GGML_TENSOR_FLAG_OUTPUT` and
> **nothing else**. An input tensor is therefore only guaranteed to hold its value for one
> execution: from the second on, a later node's scratch may be sitting on top of it.

On this graph it is not hypothetical. `dit.cond_proj` consumes the T5 states in the third
node, and reading that tensor back after one execution shows it fully overwritten — the
first forward is right and every one after it is quietly wrong. Symptomatically the
inversion looked perfect (it is one forward per step at terry's settings, and the first is
always right) while the generation pass came out anti-correlated with the reference.

`DitRunner::predict` therefore re-uploads **every** input before **every** execution,
including the conditioning and the position indices that never change. It costs ~50 KB of
transfer per forward. `tests/graph_reuse_test.cpp` asserts that contract and reports, per
backend, what actually happens to an input that is not re-uploaded — survival is
graph-dependent, so that half is printed rather than gated. Phase 4's KV cache will lean on
the same pattern much harder.

## Measured parity

30 s stereo (750 latent frames), terry's exact settings, the same noise stream on both
sides, against **torch on CPU**. RTX 5070 Laptop, F32 unless noted.

| checkpoint | cossim | max abs err |
|---|---|---|
| normalized prompt latent (encode → sample → normalize) | 1.0000000 | 2.7e-05 |
| latent at the pivot (75 regularized inversion forwards) | 1.0000000 | 3.9e-04 |
| edited latent (50 guided generation forwards) | 0.9999687 | 2.8e-01 |
| decoded audio | 0.9999179 | 5.8e-02 |

with rms-envelope 0.9999962 and log-spectrum 0.9999646 on the audio.

The inversion is exact and the generation is where it drifts, which is the expected shape:
guidance combines two forwards as `5·cond − 4·uncond`, so each step amplifies whatever
disagreement the two branches carry, 25 times over.

### Against what terry actually runs

torch on CUDA and torch on CPU are not the same function here — audiocraft's cross-attention
takes the xformers path on GPU — and the gap is larger than ours:

| | vs torch CPU | rms-envelope | log-spectrum |
|---|---|---|---|
| **torch CUDA** (terry in production) | 0.9978727 | 0.9999059 | 0.9991975 |
| audiocraft.cpp F32 | **0.9999179** | 0.9999962 | 0.9999646 |
| audiocraft.cpp F16 | 0.9978592 | 0.9997992 | 0.9981653 |

So audiocraft.cpp at F32 sits closer to torch-CPU than terry's own GPU output does, and at
F16 it sits about as far away as terry does. This is the honest framing for "does it sound
the same": F16 is a legitimate tier, not a compromise, because the reference itself moves by
that much when it changes device.

Chasing this is also what makes a divergence findable at all. A single forward is the
smallest thing that can disagree, so start there — `--trace-forwards` on the dumper writes
the input, flow step and output of the first N DiT calls, and `ac-dit` can be pointed at
exactly those.

## Speed

30 s window, 125 DiT forwards, RTX 5070 Laptop 8 GB. Solve only; model loading excluded.

| | inversion (75) | generation (50) | total | per forward |
|---|---|---|---|---|
| audiocraft.cpp F32 | 15.19 s | 10.07 s | 25.3 s | 202 ms |
| audiocraft.cpp F16 | 7.50 s | 4.88 s | 12.4 s | 99 ms |
| torch CUDA (xformers) | — | — | **6.6 s** | 52 ms |

**We are about 2x slower than torch here**, and that should be said plainly: the value
delivered so far is removing Python, not speed. torch is running xformers' memory-efficient
attention and cuDNN kernels against our straightforward `ggml_soft_max_ext` path, and none
of the obvious levers have been pulled yet — flash attention, batching the two guidance
branches into one graph the way audiocraft does, or keeping the DiT resident instead of
reloading the codec for the decode. Wall clock for the whole `mf-edit` including three model
loads is 28.2 s at F32 and 16.6 s at F16.

The regularized inversion costs 1.5x what the generation does per step, for exactly the
reason `flow_forward_count` says: 3 forwards per step against the generation's 2.

## The conditioning is shared between both branches

Classifier-free guidance's null branch does not need its own text encode. Dropout removes
the text, `T5Conditioner.forward` zeroes both the embeddings and the mask, and the DiT then
masks every real key to −inf — so cross-attention returns the zero value vector and the
residual passes through untouched, whatever the context held. `DitRunner` keeps one
conditioning tensor and switches only the mask, which is verified: feeding 1, 4 and 13 tokens
through the null branch gives bit-identical output.

The zero-attention key is what makes that defined at all. audiocraft pads k and v with one
zero timestep at the *front* and pads the additive mask with a zero column at the front to
match; drop it and the null branch's softmax is over nothing but −inf.

## Reproducing

```bash
# reference (needs torch and audiocraft -- use terry's venv). --device cpu matters:
# torch's CUDA path takes the xformers branch and is the outlier, see above.
PYTHONPATH=.../services/melodyflow .../env/Scripts/python.exe tools/dump_mf_edit_refs.py \
    --input refdata30/input48k.wav --out refedit30 --device cpu

# ours, replaying the reference's noise
mf-edit --dit models/melodyflow-dit-t24-1.0B-v1.0-F32.gguf \
        --codec models/melodyflow-vae-48khz-v1.0-F32.gguf \
        --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf \
        --input refedit30/mf_edit_input.wav \
        --prompt "Lively accordion music with a European folk feeling" \
        --noise refedit30/mf_edit_noise.f32 \
        --out-prompt-latent cppedit30/mf_edit_prompt_latent.f32 \
        --out-intermediate cppedit30/mf_edit_intermediate.f32 \
        --out-latent cppedit30/mf_edit_latent.f32 \
        --out cppedit30/mf_edit_audio.f32

python tools/cossim.py --ref refedit30 --cpp cppedit30
```

To bisect, shorten first: `--seconds 4 --steps 5` on both sides runs in seconds, and
`--no-regularize` separates the solver loop from the KL term.

## Remaining integration gates

- **The service is `melodyflow-server`** ([docs/SERVICES.md](SERVICES.md)); the presets
  from `variations.py` live there rather than in `mf-edit`, which takes a raw prompt and
  flow step.
- **The 34 presets from `variations.py` are not carried over.** `mf-edit` takes a raw prompt
  and flowstep; the preset table belongs with the server.
- **Speed.** 2x behind torch, with the levers listed above untouched.
- **Peak memory.** The codec's F32 im2col wants 2.5 GB to encode and 3.8 GB to decode 30 s,
  which is why the models are staged one at a time. Tiling is the fix if a smaller card
  needs to run this.
- **Quantized tiers.** F16 is measured; Q8_0 and below are Phase 6, and the perceptual
  measures in `tools/cossim.py` (`rms-envelope`, `log-spectrum`) are there to judge them.

## Deferred

- **A conditioned inversion.** `src_descriptions` is `[""]` in terry and the inversion is
  therefore purely unconditional. `mf_edit_latent` hard-codes that; supporting a real source
  prompt is a parameter, not a redesign.
- **Batching the two guidance branches into one graph.** audiocraft does it and calls it 2x
  faster. Here it means a graph built for batch 2 and a second one for batch 1, or accepting
  a wasted half on the inversion.
- **Seed parity with torch.** Out of scope; see above.
