# Shared DoRA-rows adapters

The shared adapter layer (`src/ac/lora.h`) supports additive `lora` and
`dora-rows`. MelodyFlow exposes both through `mf-train --adapter-type` and
`mf-edit --lora`. MusicGen exposes `mg-train --adapter-type` and
`mg-generate --lora`, with the same adapter layer used by both HTTP services.
See [MUSICGEN_LORA.md](MUSICGEN_LORA.md) for full-sequence training and the shared
smart-dice prompt-pool contract. Codebook ordering belongs to the MusicGen training
objective, not the DoRA weight transformation.

## Weight transformation

For each output row, with frozen base weight `W`, rank factors `A/B`, and
`s = alpha/rank`, the unit-strength transformation is:

```text
V = W + s * B@A
W_dora[row] = magnitude[row] * V[row] / sqrt(sum(V[row]^2) + input_dim*1e-12)
```

Magnitude starts at the stabilized base row norm; `B` starts at zero and `A`
uses the same random initialization as ordinary LoRA. The norm remains fully
differentiable, following SA3's functional DoRA-rows training formulation. It
is not a detached-denominator approximation.

The implementation factors the row norm into base, cross and low-rank Gram
terms. Neither training nor inference constructs a dense adapted matrix. Frozen
F16 weights stay F16 for the base matmul and norm cross term; adapter parameters
are F32. F16/F32 base projections are supported; quantized DoRA bases are not
supported in this implementation. Inference currently recomputes the factored
norm in its functional graph rather than caching the final row scale.

For DoRA, `--lora-strength k` interpolates the complete effective weight:
`W_eff = W + k*(W_dora-W)`. This includes magnitude adaptation, ensures strength
zero is the base, and extrapolates for strengths outside `[0,1]`. At strength
one this is the DoRA transformation above. This strength convention differs
from scaling only the low-rank term inside DoRA normalization. Additive LoRA's
existing strength behavior is unchanged.

## Adapter files and API

The GGUF has `ac.lora.type = lora|dora-rows`, rank/alpha/version metadata, and
per-target `.lora_A`, `.lora_B`, plus `.magnitude` for DoRA. Missing type metadata
means ordinary LoRA, preserving older files. The format version remains 1.
Loaders validate type, shapes, complete tensor sets, finite parameters and base
name when supplied. The architecture identifier is `ac-melodyflow-lora` for
MelodyFlow. `load_lora` also accepts generic `ac-lora` files with arbitrary
matrix target names; `load_mf_lora` separately restricts MelodyFlow targets to
supported attention projections. These are model-specific adapter parameters,
not portable SA3 checkpoint files.

Magnitude parameters and optimizer moments participate in training's existing
global gradient clipping/Adam update. Checkpoints contain adapter weights only;
optimizer resume remains unsupported. Norm constants are restored each training
step because the graph allocator may reuse their input buffers. Inference norm
buffers have shared ownership in the adapter map. The DiT inference graph
reserves additional nodes for functional adapter branches.

## Matched experiment

The first trial uses ordinary flow matching and the same full Koan cache as the
ordinary logit-normal adapter: 136 training / 32 validation windows, 30 seconds,
rank/alpha 32, attention blocks 12–23, 500 steps, learning rate 1e-4, caption
dropout .1, seed 1234, logit-normal Gaussian mean 0/stddev 1. Only the adapter
formulation changes. Magnitude initialization consumes no RNG draws, preserving
the A initialization, sample order, posterior/noise draws, caption dropout and
training times. Validation remains uniform/stratified.

The reproduction script is `build-cuda/arms-folded-koan/run-dora-study.py`;
provenance and outputs are in `build-cuda/koan-dora-study/`. Listening compares
base, ordinary logit-normal LoRA and DoRA on Bluey (30 seconds) and the newer Gary
input (21.96 seconds), at .11/.08/.05/.03, using the same dataset caption, CFG 4,
generation-only strength 1, and inversion regularization 2/keep 1/KL .2.
Inversion states must match the base exactly. All clips are globally level-matched
without clipping.

## Verification

Tests cover full tiny-DiT finite-difference gradients including magnitude,
fixed-sample loss reduction, frozen base weights, save/load agreement,
generation-only inversion identity, malformed type/magnitude rejection,
checkpoint sampling independence, and generic loader use. The functional
projection is also checked against an independently materialized weight matrix
at strengths 0/.5/1/1.4/-.5. Targeted tests pass in CPU, CUDA and Vulkan builds.

The installed Gary4Local PyTorch FlowModel provides an independent forward,
gradient and three-step Adam oracle, with explicit materialized DoRA
parametrizations. All ten cases pass for CPU/F32 and CUDA/F16 frozen weights:
conditional, padded, null, bridge and bridge-null, each with zero/nonzero B and
nontrivial magnitudes. Worst gradient relative L2 was 2.14e-6 (CPU/F32) and
1.67e-6 (CUDA/F16). These are small-model checks, not full-1B PyTorch training
parity. The full CUDA run separately checks practical memory use and saved
adapter application.

## First trial results

The full 500-step CUDA run completed in 322.37 seconds after its initial probe.
Its training graph reserved 4350.5 MiB. The initial validation loss exactly
matched ordinary LoRA at 1.0536138. At 250 steps DoRA was 1.0527567 versus LoRA
1.0527332; at 500 DoRA was 1.0519446 versus LoRA 1.0518906. These differences
do not demonstrate a DoRA advantage on the fixed velocity probe.

Training logs confirm identical sample order across all 500 steps. The saved
adapter has 180 finite F32 tensors: 60 A/B pairs and 60 magnitude vectors, versus
ordinary LoRA's 120 A/B tensors. The normalizer is differentiated during actual
training; magnitude participates in Adam and is serialized, not an inference
postprocessing substitute for trained DoRA.

All 16 new edit renders completed, covering both inputs and four pivot times for
each adapter. Inversion states exactly match the base in every render. The
listening matrix includes eight reused base renders plus both input references:
26 globally level-matched stereo 48 kHz WAVs, all verified for duration and peaks
below clipping. Whether DoRA sounds more useful than LoRA remains a listening
judgment. This is one matched seed/recipe, not a general comparison of the two
methods. Open `build-cuda/koan-dora-study/listen.html` for the complete matrix;
`results.json` contains the commands, metadata and validation results.
