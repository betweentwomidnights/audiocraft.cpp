# SA3 training ideas for AudioCraft adapters

Reviewed the local `C:/dev/sa3.cpp` native training sources on 2026-10-07,
against `src/mg/train.h`, `tools/mg-train.cpp`, and the cache preparation tools.
This is a source audit and experiment proposal; it does not change trainer defaults.

The feasible shared controls have since been implemented for both trainers. See
[TRAINING_CONTROLS.md](TRAINING_CONTROLS.md) for warmup, accumulation, tiled whole-track
coverage with fresh crops, AdamW controls, optional FFN targets, null-text probes,
and exact resume. The table below records the original pilot comparison.

## Objective matters more than transformer architecture

MusicGen predicts discrete EnCodec tokens with causal next-token cross-entropy.
There is no diffusion time, noisy latent, or timestep SNR in this training graph.
Min-SNR weighting therefore has no direct MusicGen counterpart. The original
[Min-SNR paper](https://arxiv.org/abs/2303.09556) weights diffusion timestep losses;
applying an arbitrary token weighting under that name would change the objective
without preserving its rationale.

SA3's inspected native trainer has truncated logit-normal timestep sampling and
distribution shift, including a `LogSNR` shift option. That option is not a
Min-SNR loss-weighting switch. Its loss is velocity MSE, optionally weighted by
inpainting region and padding. No explicit Min-SNR implementation was found in
the inspected native training sources.

MelodyFlow is a continuous flow model, so timestep sampling/weighting experiments
are relevant there. However SA3 uses `x_t=(1-t)z+t*n`, target `n-z`, and a flipped,
truncated logit-normal sampler. Any port must respect MelodyFlow's own time and
velocity conventions. A DDPM Min-SNR formula should not be copied unchanged.

## Things that transfer

| Feature | SA3 native trainer | MusicGen pilot | Suggested use |
| --- | --- | --- | --- |
| DoRA rows | Default; magnitude and direction differentiated | Already implemented and oracle-tested | Keep baseline |
| Caption dropout | 0.1 | 0.1 | Keep; separately score null-conditioning validation |
| Global gradient clipping | 1, after accumulation averaging | 1, batch one | Keep ordering when adding accumulation |
| Learning rate | InverseLR with exponential warmup | Constant 1e-4 | Add configurable warmup/schedule |
| Optimizer | AdamW; configured beta2=0.95, decay=0.01 | Adam; beta2=0.999, no decay | Expose settings, test separately |
| Effective batch | Gradient accumulation via batch size | One window per update | Try 4 windows/update without increasing graph memory |
| Data coverage | Full-track pre-encode, fresh random latent crops | Four fixed 12-second windows per track | Cache full-track codes, draw fresh random token crops |
| Adapter targets | Full/core scope plus include/exclude | Final 12 layers, attention only | Expose scope; test FFN projections as a separate ablation |
| Resume | Adapter plus optimizer/RNG/traversal state | Weight checkpoints only | Add exact resume, not just adapter reload |

Sources in SA3: `src/train_config.h`, `src/train_optimizer.h`,
`src/train_loop.h`, `src/train_job.h`, `src/train_latents.h`,
`src/train_diffusion.h`, `src/train_lora.h`, `src/train_resume.h`.

SA3's default warmup starts at 0.005 of the nominal learning rate and reaches
about 0.714 at update 250 and 0.918 at update 500. Its inverse decay with
inv_gamma=1e6 is negligible for these short pilots. Copying the entire preset
would change the cumulative learning-rate exposure considerably; compare with
that fact recorded rather than interpreting it as an architectural advantage.

Weight decay also deserves parameter groups. A DoRA magnitude starts at the base
weight's row norm; decaying it toward zero shrinks the effective base projection,
even with B=0. Start any decay ablation with magnitude decay disabled, and make
A/B decay configurable. SA3's inspected optimizer applies its configured decay
to all supplied parameter vectors, including magnitude.

Full-track EnCodec encoding needs its own parity checks: do not assume SA3's
continuous-latent padding/RMS rules apply to integer codec codes. Crop real frames,
then construct the delayed codebook pattern for that crop; never crop an already
delayed sequence. Preserve track-level train/validation separation. Codec-space
random crops cannot eliminate the crop's causal start boundary, but broaden the
audio seen during repeated passes and avoid repeatedly re-encoding audio windows.

## Proposed next experiments

1. Keep generation CE, rank 32, attention targets and existing data fixed. Test
   warmup alone against the constant-rate pilot, saving intermediate checkpoints.
2. Add accumulation of four windows, reporting optimizer updates and total windows
   seen separately. Compare equal data exposure; 500 accumulated updates see four
   times as many windows as the current 500-update run.
3. Expand to full-track code caches and fresh random crops. Keep held-out evaluation
   crops fixed for comparability, with both captioned and null-conditioning CE.
4. Test AdamW settings and broader targets independently after those baselines.

The current corrected pilot's 250-step generation checkpoint is effectively tied
with the base in held-out CE; 500-step checkpoints worsen it. This is consistent
with excessive fitting/update exposure, but does not identify its cause. Listening
favored generation training over continuation-masked training. Preserve both
fixed-seed unconditional and captioned listening checks: likelihood and perceptual
style adaptation answer different questions.

Across Gary models, share adapter/prompt-pool packaging, frozen-base checks,
optimizer controls, reproducibility and evaluation tools. Keep each model's
training objective and conditioning contract explicit.
