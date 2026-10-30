# MelodyFlow LoRA training audit (2026-10-05)

The trainer passes an independent PyTorch backward and optimizer comparison.
The weak editing effect is not evidence of a broken LoRA backward pass. This is
still an experimental domain-adaptation trainer, with no supervision for
transferring style while preserving a source song.

## Verified mechanics

`tests/mf_train_torch_parity.py` constructs a small **installed AudioCraft FlowModel**,
with six transformer blocks, RoPE, two skip projections, zero attention, conditioning
projection, and exact erf GELU. PyTorch parametrizations add the same LoRA matrices
to self-attention and split cross-attention projections. It compares all 60 A/B
gradient tensors and three clipped Adam updates against `mf_train_parity_probe`,
with identical weights and inputs. Cases cover conditional, padded and null text,
starting with both zero and nonzero B. The oracle executes AudioCraft's forward,
rather than reproducing our graph in Python.

| Frozen weights / native backend | Worst gradient relative L2 | Worst Adam parameter absolute difference |
| --- | ---: | ---: |
| F32 / CPU | 1.21e-6 | 1.43e-6 |
| F32 / CUDA | 1.42e-6 | 2.83e-6 |
| F16 / CUDA | 1.11e-6 | 2.35e-7 |

The F16 reference uses identical quantized weights with F32 arithmetic. These are
small-model checks, not full 1B PyTorch training parity. Earlier full-model tests
established forward/edit parity. Routine LoRA and CLI tests pass in CPU, CUDA and
Vulkan builds, including the newly added sampler checks.

Normalization and posterior sampling match the installed runtime:
`(z - latent_mean) / (latent_std + 1e-5)` and
`z = mean + (softplus(scale) + 1e-4) * noise`. Across the 136 Koan training caches,
posterior mean RMS is 1.15995 and posterior stddev RMS is 0.04875 after normalization;
posterior noise does not dominate these examples. Our interpolation and target are
`z_t = t*x + (1-t)*epsilon` and `v = x - epsilon`, agreeing with the
[published objective](https://arxiv.org/html/2407.03648v2#S2.SS2).

## Recipe differences and adapter coverage

All listening adapters so far used uniform training time. The
[paper](https://arxiv.org/html/2407.03648v2) specifies logit-normal training sampling,
uniform validation, and optimal-transport minibatch coupling. Its reported dropout
is 20%; the released checkpoint instead configures 10%. Our trainer matches that
checkpoint's dropout but uses independent batch-one noise/data pairs. That is a
valid flow-matching objective, not a reproduction of the complete base recipe.
Gradient accumulation alone would not provide coupling: examples must be paired
together before their gradients are accumulated.

`mf-train` now supports `--t-sampling logit-normal --t-mean 0 --t-std 1`.
Mean 0/stddev 1 are explicit experimental parameters; the inspected checkpoint
config does not expose them. Uniform remains the default for reproducibility.
Validation stays uniform/stratified regardless of training sampling. The sampler
does not change the velocity target or input/output scaling.

Our optimizer is Adam with beta2 0.999, clip 1, constant learning rate, no weight
decay and no EMA. These are experimental adaptation settings. They differ from
the paper's AdamW recipe; LoRA need not copy full pretraining's optimizer settings.

The Koan run adapts attention in blocks 12–23, rank 32, alpha 32. Feed-forward,
skip, timestep, conditioning and input/output projections stay frozen. Higher rank
expands existing projections; it cannot adapt omitted operations. Null conditioning
masks real text keys, leaving only a zero key/value. The oracle confirms zero
cross-attention LoRA gradients for null-only examples; self-attention still learns.

## Editing implications and next experiments

MelodyFlow learns a generative objective and edits via inversion at inference;
edit-specific training is not required for basic correctness. Our adapter likewise
learns from Koan audio/captions and ordinary noise mixtures. It never sees a
source-song/Koan-style target pair, or regularized inversion states of a source
song. Held-out velocity MSE alone cannot establish useful style transfer or
preservation of melody and rhythm.

Generation-only correctly uses frozen-base inversion and adapted regeneration.
Pivots match prompt-matched controls exactly. That changes the forward field
after a base-model inversion: a training/inference distribution mismatch worth
investigating, not a confirmed bug. Both-pass editing remains an available control
and was included in earlier listening matrices.

At guidance `g`, the adapter's guided velocity change is
`(1+g)*delta_v_cond - g*delta_v_null`. A similar change in both branches is not
amplified by CFG. Varied prompts need not amplify a shared style shift. This is an
algebraic implication, not a measured diagnosis of the clips.

First isolate logit-normal sampling at the same rank/coverage and modest step
count. Then compare broader layer/operation coverage with higher rank, retaining
fixed listening prompts, noise and base controls. An edit-aware follow-up could
use reflow-style training from **frozen-base inverted Koan latents to their own
Koan targets**, evaluated with reconstruction and held-out edit checks. Those
endpoints share content; arbitrarily pairing Bluey with an unrelated Koan song
would instead teach unwanted content replacement. This remains a proposal, not
an established style-transfer objective. The subsequent
[edit-bridge experiment](MELODYFLOW_EDIT_TRAINING.md) implements and tests this
pairing as a separate experimental objective, with matched noise controls.

## Reproduction

The opt-in oracle needs an installed AudioCraft/PyTorch runtime. It does not modify
the application or make native training depend on PyTorch:

```powershell
& "$env:APPDATA/Gary4JUCE/services/melodyflow/env/Scripts/python.exe" tests/mf_train_torch_parity.py --runtime "$env:APPDATA/Gary4JUCE/services/melodyflow" --gguf-path .venv/Lib/site-packages --probe build-cuda/bin/Release/mf_train_parity_probe.exe --out build-cuda/training-oracle-new --device gpu --weight-type f16
```

Use an empty output directory. Detailed results and inspected checkpoint config
are in `build-cuda/koan-training-audit/`. No long retraining was started in this audit.
