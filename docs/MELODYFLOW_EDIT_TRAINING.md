# Experimental MelodyFlow edit-bridge LoRA

This is an experimental training objective for reconstruction from frozen-base
inversion states. It is not a paired cross-style dataset and does not establish
that the adapter preserves a new song's content while transferring Koan style.
The ordinary flow-matching trainer remains the default.

## Content-aligned bridge

Take a normalized, sampled Koan latent `x`, invert it with the frozen base model
to a pivot `p` at physical flow time `s`, and retain the same `x` as the target.
At physical model time `t` in `[s, 1]`, the straight bridge is:

```text
u = (t - s) / (1 - s)
z_t = (1 - u) * p + u * x
velocity_target = (x - p) / (1 - s)
```

The model receives physical `t`, not local bridge fraction `u`. Its timestep
embedding, input variance scaling and output scaling stay intact. The interval
length appears in the velocity target because the solver integrates in physical
time. `MfTrainGraph::compute_bridge` expresses this exactly through the existing
graph using the equivalent time-zero endpoint `(p - s*x)/(1-s)`.

That endpoint is generally not independent Gaussian noise. Learning from this
pairing is the intended difference from ordinary flow matching. The straight
bridge approximates a trajectory; it is not a recording of every base ODE state,
and a lower bridge MSE need not improve the actual ODE reconstruction endpoint.
The study measures both separately.

## Preparation and training

`mf-edit-cache --mode prepare` reads the existing normalized posterior/T5 cache,
selects its first sorted window per source track, samples its posterior once, and
performs frozen-base inversion. It keeps the original train/validation split and
rejects source leakage. Inversion uses empty conditioning, Euler 25, regularization
2/keep 1, KL 0.2 and sway -0.8, matching our listening edit configuration.

It writes two cache trees with **identical fixed targets and captions**:

- `control`: ordinary training caches with posterior stddev zero. Each training
  step pairs the fixed target with newly sampled independent Gaussian noise.
- `bridge`: edit caches with the fixed target, saved inversion pivot and physical
  pivot time. Training uses their content-aligned pairing.

Ordinary cache validation now allows zero posterior stddev for deterministic
targets, while rejecting negative/nonfinite values. Edit caches require fixed
targets and a valid pivot. Noise draws are still consumed by both training paths,
so matching seed, files and time sampling preserves shuffled order, caption dropout
and training times between the control and bridge runs.

```powershell
build-cuda/bin/Release/mf-edit-cache.exe --mode prepare --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --cache build-cuda/koan-30s-cache --out build-cuda/koan-edit-supervision/paired-cache --frames 750 --flowstep 0.11 --seed 1234 --device gpu

build-cuda/bin/Release/mf-train.exe --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --cache build-cuda/koan-edit-supervision/paired-cache/bridge/train --validation build-cuda/koan-edit-supervision/paired-cache/bridge/validation --out build-cuda/koan-edit-supervision/edit-bridge-500.gguf --frames 750 --rank 32 --alpha 32 --first-layer 12 --steps 500 --lr 0.0001 --cfg-dropout 0.1 --seed 1234 --device gpu --eval-samples 0 --eval-draws 3 --checkpoint-every 250 --t-sampling logit-normal --t-mean 0 --t-std 1 --t-min 0.11 --objective edit-bridge
```

For the matched noise control, use the `control` cache tree, another output path,
and `--objective flow-matching`, retaining `--t-min 0.11`. Logit-normal samples a
local fraction, mapped onto `[0.11, 1]` in both runs. Validation instead uses fixed
stratified local fractions. Explicit `--t-min` must match the pivot time of every
edit cache; without it the edit objective uses each cache's own pivot time.

`mf-edit-cache --mode render` integrates from a saved pivot to time 1, with the
cached caption, optional adapter and chosen CFG. Its output is normalized planar
float latent data. This enables endpoint reconstruction against the exact cached
target without another codec encoding or another inversion/noise draw. Use the
cached normalization constants to denormalize before decoding with `ac-vae`.

## Study and verification

The full-data sampling comparison reuses uniform 500 steps and trains logit-normal
500 steps on the original 136 training / 32 validation windows. Logit-normal
mean 0/stddev 1 remain experimental parameters. The fixed validation probe is
1.0536138 for the base, 1.0508215 for uniform at 500, and 1.0518906 for logit-normal
at 500. This single trial did not show a clear logit-normal advantage.

The matched edit pilot uses one fixed 30-second window per track: 34 training and
8 held-out tracks. Both adapters start fresh with rank/alpha 32, blocks 12–23,
500 steps, learning rate 1e-4, dropout 0.1 and seed 1234. Compare the bridge against
its matched noise control; comparisons with the full-data adapters also change
the number of windows, target resampling and time interval. Absolute validation
losses from different objectives are not directly comparable.

Listening uses Bluey's first 30 seconds, flowstep 0.11, the same recorded PyTorch
noise, generation-only strength 1, and both the untouched Koan dataset caption
(CFG 4) and empty text (CFG 0). Inversion latents must match the base exactly.
Raw float audio is retained and WAVs are globally level-matched without clipping.
Held-out ODE reconstruction uses identical cached pivots/targets and CFG 4 for the
base, matched control and bridge adapter; latent endpoint MSE is a shared metric.

Both matched adapters completed 500 steps. The bridge's fixed held-out velocity
probe fell from 0.3090563 to 0.0330819; the noise control's own probe changed from
0.9988766 to 1.0007338. Their absolute losses measure different targets. Training
logs confirm identical source-window order across all 500 steps.

Actual ODE endpoint MSE across all eight held-out tracks, using the same pivots,
targets, captions and CFG 4, was:

| Generation model | Mean normalized latent MSE |
| --- | ---: |
| Frozen base | 0.66830098 |
| Matched noise control | 0.60630482 |
| Edit bridge | 0.37535819 |

The bridge reduced mean endpoint error by 38.1% versus the matched control and
43.8% versus the base. It improved over the control on every held-out track
(individual reductions 28.9–54.5%). This supports the content-aligned objective
for reconstruction, not a claim of better cross-style musical edits. Rank stayed
at 32 to isolate the training objective. The study uses one seed and one pivot
time; generalization to other inputs and times remains open.

All ten Bluey comparison WAVs and eight reconstruction listening WAVs were
verified as finite 30-second stereo 48 kHz audio and level-matched without
clipping. The first two held-out examples include decoded targets, base,
control and bridge outputs. The Bluey matrix includes base, uniform/full,
logit/full, matched control and bridge for each prompt condition.

The new physical-time bridge loss is checked against a hand-derived inference
prediction and independent installed PyTorch FlowModel gradients/Adam updates,
including null conditioning and F16 CUDA weights. The CLI tests exercise cache
preparation, identical target pairing, rejection of mismatched cache objectives
and pivot times, checkpoint sampling independence, and adapted reconstruction.
The targeted tests pass in CPU, CUDA and Vulkan builds. CPU F16 matmul arithmetic
does not meet this oracle's tight dequantized-F32 tolerance; CPU oracle checks use
F32 weights. These are small-model backward checks, not a full 1B PyTorch trainer.

Scripts and provenance are in `build-cuda/koan-edit-supervision/`:
`results.json`, training/preparation logs, `paired-cache/`, float outputs,
level-matched WAVs and `listen.html`. Reproduction scripts are
`build-cuda/arms-folded-koan/run-logit-trial.py` and `run-edit-study.py`.

## Multiple pivot times and inversion regularization

Preparation also accepts `--windows-per-source N` and a comma-separated
`--flowsteps` list. Within each source, sorted windows are assigned cyclically
to that list; each window gets its own posterior/inversion seed. This selects
one pivot per window rather than preparing every window at every pivot.
The default remains one window at `--flowstep 0.11`.

```powershell
build-cuda/bin/Release/mf-edit-cache.exe --mode prepare --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --cache build-cuda/koan-30s-cache --out build-cuda/koan-multipivot-study/paired-cache --frames 750 --device gpu --seed 4321 --windows-per-source 4 --flowsteps 0.03,0.05,0.08,0.11
```

Both prepared cache trees store `ac.cache.time_min`. The noise control uses that
per-example lower time bound automatically, matching the bridge's physical time
sampling with identical seed/order/dropout. Explicit `--t-min` must agree with
each cache; omit it when training a mixed-pivot dataset. Old ordinary caches
without this key retain the existing default of zero. Metadata changes do not
alter the objective's target velocity.

The broader pilot keeps rank/alpha 32, blocks 12–23, 500 steps, learning rate
1e-4, dropout 0.1, logit-normal sampling and regularization 2/keep 1. It uses
four windows per track, 136 training/32 validation, and seed 4321. Window coverage,
pivot coverage and stochastic samples change together, so this study does not
attribute any gain to one of them individually. A matched noise control and the
prior bridge are compared on identical held-out pivots. Endpoint checks select
one window from each of the eight held-out tracks, balanced two per pivot time.

Inversion regularization is separate from LoRA/weight regularization. In this
implementation `edit_inversion_params` enables it, while
`edit_generation_params` explicitly disables it. Inner iterations refine the
inverted state; retained iterations include a KL correction against velocities
from a noisy flow-path reference. Thus the settings affect forward editing
indirectly through its starting latent. More iterations are not a guarantee of
stronger style transfer or higher fidelity.

The separate `.05` Bluey ablation uses the prior adapter and dataset caption:
2 iterations/keep 1, 4 iterations/keep 1, and no regularization, each with a base
control. Keeping one corrected iterate lets 2 and 4 reuse the same 51 recorded
noise draws. The unregularized run only consumes the initial posterior draw.
All six clips are level-matched. Relative endpoint latent changes versus 2
iterations were 0.15055 (4, bridge) and 0.56305 (off, bridge); these measure change,
not audio quality. Training pivots retain 2/keep 1 to isolate the broader dataset.

The broader study and regularization outputs are in
`build-cuda/koan-multipivot-study/`; its reproduction script is
`build-cuda/arms-folded-koan/run-multipivot-study.py`.

### Broader pilot results

The control's fixed velocity probe changed from 1.0166526 to 1.0166069;
the bridge's own probe fell from 0.4142300 to 0.0434128. The logs confirm identical
source-window order across all 500 training steps. The two objectives' absolute
velocity losses remain incomparable.

Actual endpoint reconstruction on the eight held-out windows gave:

| Pivot / cohort | Base | Prior .11 bridge | Matched noise control | Multi-pivot bridge |
| --- | ---: | ---: | ---: | ---: |
| .03 (2 tracks) | 1.284809 | 1.888983 | 1.642734 | 0.706614 |
| .05 (2 tracks) | 1.006390 | 0.823697 | 1.333738 | 0.536461 |
| .08 (2 tracks) | 1.059575 | 0.668571 | 1.044557 | 0.544059 |
| .11 (2 tracks) | 0.851549 | 0.535241 | 0.769356 | 0.669538 |
| All 8 | 1.050581 | 0.979123 | 1.197596 | 0.614168 |

The multi-pivot bridge reduced mean endpoint MSE by 37.3% versus the prior bridge,
48.7% versus the matched control and 41.5% versus the base. It improved over the
prior bridge on all six low-pivot examples, but worsened on both .11 examples
(16.4% and 39.6% higher error). This indicates a useful lower-pivot reconstruction
gain with a tradeoff at .11 in this pilot, not uniform superiority. Each pivot
cohort has only two examples and uses different source tracks. These comparisons
are descriptive, not a statistical estimate of generalization.

The fresh broader run also changes seed, windows and inversion samples relative
to the prior adapter. It does not isolate pivot coverage as the sole cause of
the improvement. Reconstruction error does not establish musical style transfer
quality on Bluey. The listening matrix compares both prompts at .03/.05/.08 with
base, prior bridge and matched-control outputs using identical inversion states.

All 24 main listening clips were level-matched together; the six regularization
clips were separately level-matched. All 30 WAVs passed stereo 48 kHz/30-second
format and peak checks. `results.json` records the commands, measured losses,
per-track endpoint errors, noise path and sample-order verification; `listen.html`
provides the complete audio matrix. Finalization and checks use
`build-cuda/arms-folded-koan/finalize-multipivot-study.py`.
