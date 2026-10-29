# Shared adapter training controls

Implemented for both `mg-train` and `mf-train`. Existing defaults remain constant
learning rate, one window per update, Adam beta1/beta2=0.9/0.999, epsilon=1e-8,
global clipping=1, no weight decay, and attention targets only.

## Data coverage

Both preparation scripts accept `--cache-seconds 30 --cover-track`. This tiles each
source with native codec windows up to 30 seconds, anchoring the final window at
the track end. It avoids uncovered tails and does not pad short tracks: sources
shorter than the requested cache window fail explicitly. Choose a shorter cache
window for such datasets. Track splitting still happens before encoding.

At training time `--frames` can be smaller than a cache's frame count. Add
`--random-crop` to draw a new uniformly distributed valid start on every training
visit. All codebooks/latent channels share the same crop. MusicGen builds its
delayed pattern **after** cropping undelayed codes. MelodyFlow crops posterior
mean/std together, then samples the posterior and noise. Caption tensors remain
associated with the source. Validation always uses deterministic center crops.
Without `--random-crop`, training uses center crops too.

This is a bounded tiled-cache implementation, not SA3's monolithic full-track
latent encoding. It retains codec boundaries at cache edges and may overlap the
last two cache windows. More cache windows also consume more host RAM: the current
trainers load their cache directory into host memory. Edit-bridge caches must
match `--frames` exactly; arbitrary cropping of paired pivots is not supported.

For example, prepare MusicGen caches with the existing required model/dataset
arguments plus `--cache-seconds 30 --cover-track`, then train with:

```powershell
build-cuda/bin/Release/mg-train.exe `
  --lm models/musicgen-small-0.4B-v1.0-F16.gguf `
  --cache CACHE/train --validation CACHE/validation --out RUN/adapter.gguf `
  --frames 600 --prefix-frames 0 --adapter-type dora-rows --rank 32 `
  --steps 500 --lr 0.0001 --warmup-steps 50 --accumulation-steps 4 `
  --random-crop --checkpoint-every 100 --device gpu
```

For MelodyFlow, use the corresponding preparation script and model arguments,
then `mf-train --frames 300 --objective flow-matching --t-sampling logit-normal`
for 12-second training crops, plus the same shared controls. These are example
experimental settings, not newly established quality defaults.

## Optimizer and evaluation

- `--accumulation-steps N`: N microbatches per optimizer update. Gradients are
  averaged **before** global clipping. `--steps`, warmup and checkpoint intervals
  count optimizer updates. Logs report total windows seen and applied learning
  rate. Compare equal windows seen when comparing batch sizes.
- `--warmup-steps N`: linear warmup, lr multiplier `min(1, step/N)`, first update
  numbered one. Zero disables warmup. This is deliberately simpler than SA3's
  exponential warmup; it does not claim to reproduce SA3's preset.
- `--lr-schedule constant|inverse`: constant is the default. Inverse additionally
  multiplies by `(1+(step-1)/1e6)^(-0.5)`, matching SA3's inverse decay parameters.
  With short runs this decay is slight; warmup is usually the larger change.
- `--beta1`, `--beta2`, `--adam-eps`, `--grad-clip`: validated Adam controls.
- `--weight-decay F`: decoupled AdamW decay of A/B only. DoRA magnitude parameters
  never decay toward zero. Zero preserves Adam behavior.
- `--train-ffn`: also adapt both feed-forward projections in the selected layers,
  in training and native inference/services. This increases memory requirements.
  `--first-layer` still selects the layer range; arbitrary include/exclude filters
  are not implemented.
- `--eval-unconditional`: replace text conditioning by the model's null path for
  fixed probes. It changes evaluation, not training caption dropout. MusicGen's
  evaluate-only modes accept it too. Compare captioned and null probes separately.

## Exact continuation

Each adapter/checkpoint now has a sibling `FILE.gguf.state`. It contains adapter
values, Adam moments, completed update count, windows seen, RNG engine, sample
permutation, and a configuration/content fingerprint of the base and caches.
The state is written through a temporary file and renamed after flushing.

Pass `--resume FILE.gguf.state --steps TOTAL` with the same training settings,
base, and cache contents, and a **new output path**. Total must exceed the saved
update count. The state includes weights, so the GGUF adapter is not separately
loaded. Changing evaluation/checkpoint frequency does not change sampling.
Changed optimizer settings, crop configuration, model/cache bytes or adapter
configuration fail validation. The fingerprint is an FNV-1a content checksum,
not a cryptographic authenticity check. State text uses round-trip float precision;
exact reproducibility is tested within the same executable/backend/runtime, not
promised across compilers or GPU architectures. Legacy adapters without state
remain usable for inference but cannot resume their lost optimizer history.

State files are larger than inference adapters; publish the GGUF and prompt pool
for serving, retain state files locally for training. There is no activation
checkpointing or streaming cache loader in this implementation.

## Verification

`train_controls` checks warmup, mean-gradient global clipping, AdamW A/B decay and
DoRA magnitude exclusion. `train_cache_windows` checks coverage and bounds.
Both trainer CLI fixtures exercise accumulation, shorter fresh crops, FFN targets,
warmup, AdamW, exact checkpoint resume, and rejected settings/configuration changes.
MusicGen also exercises null-conditioning evaluation of a broader-target adapter.
