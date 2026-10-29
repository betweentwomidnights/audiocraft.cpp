# Experimental native MelodyFlow LoRA

Both ordinary LoRA (default) and `--adapter-type dora-rows` are supported. See
[DoRA-rows support and the matched experiment](DORA_ROWS.md) for the formulation,
file format, strength behavior, verification and current model coverage.

See [the training audit](MELODYFLOW_TRAINING_AUDIT.md) for independent PyTorch
gradient/update checks, recipe differences and proposed edit-aware experiments.
The [edit-bridge experiment](MELODYFLOW_EDIT_TRAINING.md) implements a separate
content-aligned inversion-to-target objective and matched noise controls.

`mf-train` trains attention LoRAs against MelodyFlow's velocity prediction using the
shared GGML fork's autodiff. `mf-edit --lora` applies the resulting GGUF adapter.
T5 and the audio codec stay frozen and run only during cache preparation. The C++
trainer has no PyTorch dependency; the preparation script uses the existing `numpy`
and `gguf` requirements and invokes the native encoder tools.

This is a feasibility experiment. No claim is made that it improves edits or that
LoRA outperforms a full MusicGen fine-tune for continuations. MusicGen training is
not implemented in this change.

## Prepare a dataset

Each PCM WAV needs a same-stem UTF-8 `.txt` caption. Captions are data, never
instructions. The supplied datasets both fit this layout:

| Dataset | Tracks | Duration | Source PCM |
| --- | ---: | ---: | --- |
| `koan_dset` | 42 | 87.49 min | stereo 44.1 kHz, 24 bit |
| `keygen_dset` | 56 | 136.97 min | stereo 48 kHz, 16 bit |

The script handles PCM 8/16/24/32 bit and lets the native codec conform sample rate
and channels. It picks random windows across each track, encodes the VAE posterior
and T5 caption, and writes small GGUF caches. The cache holds the normalized
posterior mean/stddev, rather than one fixed posterior sample. Training redraws
both posterior noise and flow noise on each visit.

Tracks are shuffled deterministically and split **before** cropping. All windows
from one track stay in one split. `manifest.json` records source paths, captions,
crop starts, model paths and normalization fingerprint. Source files are untouched.
The trainer checks the exact normalization constants and rejects source IDs shared
between its training and validation directories. Distinct files containing the
same recording still require manual deduplication.

Build normally (`build.cmd cuda` / `./build.sh cuda`). Windows example, from the
repository root; change the executable directory for other build layouts:

```powershell
.venv/Scripts/python.exe tools/prepare_mf_lora.py --dataset C:/Users/thegr/Downloads/keygen_dset --out build-cuda/keygen-cache --bin build-cuda/bin/Release --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --codec models/melodyflow-vae-48khz-v1.0-F32.gguf --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf --seconds 2 --windows-per-track 4 --device gpu
```

Use `--max-tracks 8 --windows-per-track 2` for a quick pilot. Use an empty output
directory. Cache preparation does not resume a partially completed directory.
For Koan, change `--dataset`; the native audio conformance resamples to 48 kHz.

## Train and apply

```powershell
build-cuda/bin/Release/mf-train.exe --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --cache build-cuda/keygen-cache/train --validation build-cuda/keygen-cache/validation --out build-cuda/keygen-r4.gguf --frames 50 --rank 4 --first-layer 20 --steps 100 --device gpu

build-cuda/bin/Release/mf-edit.exe --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --codec models/melodyflow-vae-48khz-v1.0-F32.gguf --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf --input in.wav --prompt "chiptune, melodic electronic music" --lora build-cuda/keygen-r4.gguf --lora-strength 1 --out edited.wav
```

`--frames` must match the cache window: MelodyFlow uses 25 frames/s, so 2 s = 50
frames. F16 and F32 base projections are supported; adapters and activations are
F32. `--alpha` defaults to rank. The default learning rate is `0.0001`, caption
dropout is `0.1`, rank is 8, and device is CPU. CPU thread count uses `AC_THREADS`.

Training timesteps default to uniform for compatibility with earlier experiments.
`--t-sampling logit-normal --t-mean 0 --t-std 1` enables a configurable logit-normal
density. The validation probe remains uniform/stratified under either option.
The output adapter must not exist already.

Each selected layer adapts `self.qkv`, `self.out`, `cross.q`, `cross.kv` and
`cross.out`. The default trains only the last layer. `--first-layer 20` selects
the last four of T24; `--first-layer 0` selects all layers. Functional application
computes `W*x + (alpha/rank)*B*(A*x)` without forming a dense weight delta. The base
weights stay frozen. A is seeded Gaussian and B starts at zero.

The objective is mean squared velocity error with uniform `t` in [0,1]:

```
z_t = (1-t)*noise + t*normalized_audio
target_velocity = normalized_audio - noise
```

The existing DiT input rescaling and output sqrt(2) are retained. Adam updates
only A/B with global gradient clipping at norm 1. There is no weight decay.
Training uses dense attention and F32 matmul precision, independent of
`AC_FLASH_ATTN`. Padded T5 positions are masked, including during caption dropout.

GGML does not provide NORM or GELU_ERF autodiff. The training forward therefore
uses centering plus RMSNorm for LayerNorm, and the Abramowitz-Stegun erf
approximation for GELU. Inference keeps exact erf GELU and its existing LayerNorm.
Numerical tests cover these substitutions. Training also makes incoming CONCAT
and SCALE gradients contiguous; see [GGML_FORK.md](GGML_FORK.md).

The CLI prints a reproducible loss probe using fixed posterior draws, flow noise
and timesteps. By default it uses at most four validation windows and one random
timestep per window. `--eval-samples 0` evaluates every window; `--eval-draws 3`
evaluates each at stratified timesteps 1/6, 1/2 and 5/6. Evaluation has its own RNG
and does not change training sampling. Without `--validation` this is explicitly
a training probe. It is a diagnostic, not a comprehensive quality score.

`--checkpoint-every 250` saves adapters and runs the same probe every 250 steps.
For `--out adapter.gguf`, intermediate files are `adapter.step-250.gguf`, etc.
The GGUF contains adapter weights; the accompanying `.gguf.state` now preserves
optimizer and sampling state for exact resume. See [shared controls](TRAINING_CONTROLS.md).
Select checkpoints using validation and listening, rather than assuming
that the final step is best.

## Longer-window experiment

The 1B stereo model was trained on 30-second segments at 25 latent frames/s,
with an attention mask to support shorter inference durations
([paper, sections 3 and 4.3.2](https://arxiv.org/html/2407.03648v1)). Two-second
windows are useful for a mechanics smoke test. For a 30-second editing task,
30-second crops (`--seconds 30`, `--frames 750`) are the more direct starting
point: they retain longer rhythmic and melodic context. This is a hypothesis
about adaptation quality, not evidence that the longest crop always wins.

The follow-up experiment uses the full Keygen dataset, four 30-second crops per
track, 45 training tracks and 11 validation tracks. Its rank-16, last-four-layer
graph fits the RTX 5070 Laptop GPU with a 1350.3 MiB graph buffer, excluding the
frozen F16 model. A reproducible trial configuration is:

```powershell
.venv/Scripts/python.exe tools/prepare_mf_lora.py --dataset C:/Users/thegr/Downloads/keygen_dset --out build-cuda/keygen-30s-cache --bin build-cuda/bin/Release --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --codec models/melodyflow-vae-48khz-v1.0-F32.gguf --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf --seconds 30 --windows-per-track 4 --device gpu

build-cuda/bin/Release/mf-train.exe --dit models/melodyflow-dit-t24-1.0B-v1.0-F16.gguf --cache build-cuda/keygen-30s-cache/train --validation build-cuda/keygen-30s-cache/validation --out build-cuda/keygen-r16-30s.gguf --frames 750 --rank 16 --alpha 16 --first-layer 20 --steps 1000 --lr 0.0001 --cfg-dropout 0.1 --seed 1234 --device gpu --eval-samples 0 --eval-draws 3 --checkpoint-every 250
```

This trial changes dataset size, crop duration and rank together. It cannot
attribute an improvement to any one of them; a controlled duration comparison
would train otherwise identical 8-second and 30-second configurations.

Keep the original dataset captions for training and matched listening comparisons.
Use the identical caption, source audio, seed and flowstep for the base and adapter;
adding a new style prompt would confound the contribution of the adapter. A caption
can itself contain genres such as Chiptune; it is still important to compare against
the base model under that same caption.

## Validation and first result

The tiny four-layer model test exercises U-ViT skips, fused QKV views, stored RoPE,
zero-attention padding, cross-attention and the complete forward/backward graph.
It checks finite differences for A/B, exact zero-B initialization, repeatability,
fixed-sample loss decrease, frozen base weights, adapter GGUF serialization,
zero-strength inference and saved-adapter inference. Primitive tests separately
check normalization/GELU and multi-head attention derivatives. GPU inference uses
its usual arithmetic; the training graph explicitly requests F32 precision.

Initial Keygen CUDA pilot on 2026-10-05:

- Eight tracks, six training and two validation, two 2-second windows per track.
- F16 T24 base, rank 4, alpha 4, last four layers, 100 steps, learning rate 0.0001.
- 20 projection adapters; graph buffer 82.4 MiB, excluding frozen weights.
- Training loop approximately 7.9 seconds on an RTX 5070 Laptop GPU.
- Fixed held-out probe: **1.2292244 -> 1.2294386** (essentially flat).
- Adapter size: 1,281,024 bytes.

A separate one-step smoke run adapted all 24 layers (120 attention projections,
rank 2) on the same 2-second windows. Its graph used 480.2 MiB in addition to the
F16 base weights and fit on the same 8 GB GPU. All 16 registered tests passed in
CPU, CUDA and Vulkan builds, including both Intel and NVIDIA Vulkan devices.

A six-second held-out input was also edited with the saved pilot adapter. Both
baseline and adapted runs produced finite stereo 48 kHz audio. Strength zero
reproduced the baseline latent bit for bit with the same seed. Strength one
changed the latent (relative L2 difference 0.319); that measures adapter effect,
not musical quality. Listening evaluation remains outstanding.

This establishes native training and adapter serialization on the real model. It
does **not** establish a useful style adaptation. A meaningful follow-up should
use longer windows and more tracks, measure more held-out timesteps, and compare
edits on the same sources, captions, seed and flowstep at strengths 0, 0.5 and 1.
Listen for style change, preserved musical structure and codec artifacts.

The initial 30-second edit of `kk_10_25_arms_folded.mp3` at flowstep 0.13 used
the 100-step pilot and a hand-written 8-bit prompt for both versions. The listener
reported essentially no audible difference. A controlled 1000-step rerun of the
same tiny rank-4 training set made the same fixed validation probe worse:
1.2292244 before training, 1.2357711 at step 250, 1.2416005 at step 500 and
1.2555420 at step 1000. Falling training loss alone did not establish useful
adaptation. These numbers use the original four-window, one-draw probe, not the
larger three-timestep probe used for the full dataset run.

Full Keygen follow-up on the same GPU, 2026-10-05:

- 180 training and 44 validation windows, 30 seconds each; disjoint source tracks.
- Rank 16, alpha 16, last four layers, 1000 steps, learning rate 0.0001.
- Validation uses all 44 windows at three fixed stratified timesteps.
- Frozen base probe: **0.7264466**.
- Step 250: **0.7259669**; step 500: **0.7258774**.
- Step 750: **0.7256901**; step 1000: **0.7258313**.
- The lowest measured validation loss is at **750 steps**, about 0.104% below
  the base. This small difference does not establish an audible quality gain.
- Training and periodic/final probes took 480.88 seconds, excluding the initial
  probe and dataset preparation. Each adapter is 5,114,880 bytes.

The matched first-30-second listening comparison uses seed 1234, flowstep 0.13,
strength 1 and this unchanged caption from the training split:

> Electronic, Chiptune, Goa Trance, Trance. melodic, energetic, dark, space, retro, game. 144 bpm. Key of E.

Caption source: `10 Minutes of Best Keygen Sound.txt`. Outputs and experiment
provenance are in `build-cuda/arms-folded-lora/`; `full-experiment-results.json`
records the chosen adapter, command, loss probes and audio measurements. Compare
`base-dataset-caption.wav`, `pilot-dataset-caption.wav` and
`rank16-30s-dataset-caption.wav`. The rank-16 edit uses the 750-step checkpoint.
These remain listening candidates, rather than validated musical defaults.
The listener again reported essentially no audible change with this rank-16
adapter. The next experiment therefore switches to the Koan dataset for a
clearer intended style contrast and widens the adapted depth.

## Koan and generation-only editing

The Koan follow-up uses 42 tracks, split into 34 training and 8 validation tracks,
four 30-second crops per source, rank 32/alpha 32 and the last twelve T24 blocks
(`--first-layer 12`). This adapts 60 attention projections. The graph buffer is
3972.3 MiB and fits the same 8 GB CUDA GPU. Learning rate, dropout and validation
draw settings are unchanged; checkpoints are saved every 250 of 1000 steps.

`mf-edit --lora-pass generation` keeps source inversion on the base model and
uses the adapter for the forward edit. The default remains `both` for compatibility.
This makes adapter placement an explicit experiment: applying adaptation while
inverting may change the starting point and affect the eventual style transfer.
It has not yet been established that generation-only placement is better.

The listening matrix uses an untouched training caption from
`koan_sound_cobalt_92bpm_60bars.txt`:

> glitch hop, neurofunk, electronica, intricate breakbeat, 92 bpm, D minor

All edits use the same first 30 seconds, flowstep 0.13 and the exact recorded
Gary PyTorch noise stream. Compare base, generation-only strength 1,
generation-only strength 3 and both-pass strength 1. Intermediate latents are
also saved; generation-only inversion must match the base bit for bit. Higher
strength can increase artifacts as well as style change, so it needs listening.
Dataset, rank and depth are changed together here; this is a practical candidate,
not a controlled attribution of each training choice.

The fixed Koan validation probe measured 1.0536138 for the frozen base,
1.0519286 at step 250, **1.0508215 at step 500**, 1.0520197 at step 750 and
1.0517882 at step 1000. The listening matrix uses the 500-step checkpoint,
approximately 0.265% below the base on this probe. Training plus periodic/final
probes took 664.95 seconds. Each adapter is 30,679,424 bytes. The small loss gain
does not establish useful style transfer. Outputs and provenance are saved in
`build-cuda/arms-folded-koan/` (`results.json`, `train.log` and the WAV/F32 files).

All four full-length edits produced finite latents and audio. Generation-only
inversion matched the base bit for bit at both tested strengths. Relative latent
L2 changes from the base were 0.488 (generation-only strength 1), 0.938
(generation-only strength 3) and 0.549 (both-pass strength 1). These measure
change, not musical quality, and do not establish that generation-only is better.

The `*-level-matched.wav` listening files were decoded again from the saved
latents to float audio, then scaled with one constant gain per clip to RMS 0.12.
All peaks are below 0.95. This avoids the PCM clipping in the raw WAVs and makes
the comparison less sensitive to gain differences. Raw outputs are retained;
`results.json` records float peaks, listening gains and file paths.

The listener confirmed an audible effect from the Koan adapter. A follow-up
listening matrix applies the existing 1000-step checkpoint to the original
`kk_10_25_arms_folded.mp3` and the first 30 seconds of `bluey_remix.mp3`. It keeps
the same Koan caption, flowstep 0.13 and recorded noise. Each input has a base
control, generation-only strengths 1 and 3, and both-pass strength 1. These are
float-decoded and level-matched before PCM conversion. Original-input 500-step
comparisons are also included at the same playback level. Files and provenance
are in `build-cuda/koan-1000-listening/`. The 1000-step checkpoint was requested
for listening despite the lower measured validation loss at 500 steps; the
validation probe does not decide musical preference.

A Bluey flowstep sweep uses the same 1000-step adapter, generation-only strength
1, caption and recorded noise at 0.12, 0.11, 0.10 and 0.09. Each setting has its
own frozen-base control. The previous 0.13 base and adapter renders are reused
as references and level-matched together with the sweep. Lowering flowstep
changes the inversion pivot as well as the edit; paired controls help distinguish
that effect from the adapter's contribution. The script checks that inversion
latents match exactly within each base/adapter pair and that all audio is finite.
It saves raw float audio and applies one constant gain per clip before PCM
conversion, using a common RMS target constrained by a 0.95 peak ceiling.
Outputs and provenance are in `build-cuda/koan-flowstep-sweep/results.json`;
the reproduction script is `build-cuda/arms-folded-koan/run-flowstep-sweep.py`.

Empty descriptions are supported: their text tokens are masked out. For a direct
unconditional edit, use an empty prompt and `--cfg-coef 0`; this runs the null
branch alone during generation. Training's `--cfg-dropout 0.1` already gives the
adapter unconditional examples. With no text guidance, the input audio and the
adapter still influence the edit, so this does not mean generating from scratch.

The longer prompt study compares the dataset caption above, `instrumental
electronic music`, `acoustic piano, gentle melody`, and an empty description at
flowstep 0.11 on Bluey's first 30 seconds. Text prompts use CFG 4; the empty case
uses CFG 0. Each prompt has a base control and generation-only strength-1 adapter
edits at 1000 and 3000 steps. The 3000-step training run starts fresh with the
same seed, rank, learning rate, dropout, cached data and fixed validation probe;
this is not an optimizer resume. It saves intermediate adapters at 1000 and 2000
steps. The listening script checks inversion equality within each prompt-matched
base/adapter pair, finite outputs, 30-second stereo duration, and level-matches
all clips together. Files and provenance are in
`build-cuda/koan-longer-prompt-study/`; scripts are
`build-cuda/arms-folded-koan/run-longer-prompt-study.py` and
`build-cuda/arms-folded-koan/summarize-prompt-study.py`.

The restart's 1000-step adapter tensors exactly reproduce the prior 1000-step
adapter (maximum absolute difference 0), with the same validation loss of
1.0517882. The 2000-step probe is 1.0544049 and the 3000-step probe is 1.0558769,
versus 1.0536138 for the frozen base. Training and its periodic/final probes took
1642.89 seconds. Longer training did not improve this held-out conditional
velocity-MSE probe; that does not establish the adapter's audible style or its
unconditional editing quality. The comparison retains all checkpoints rather
than selecting 3000 steps as a new default based on training duration alone.

## Limits and MusicGen follow-up

Optimizer resume, gradient accumulation, random crops, and HTTP adapter selection
are implemented. Activation checkpointing and a training web UI remain future work. CLI
adapters apply to both inversion and generation by default.
`mf-edit --lora-pass generation` instead uses the base model for inversion and the adapter for generation.
This creates two DiT graph arenas sharing the same frozen weights. Model-name and tensor-shape
checks do not fingerprint every base weight: use the exact checkpoint trained
against. Test Metal separately before claiming support; CPU/CUDA/Vulkan validation
does not establish Metal training support. Reduce window length or the number of
adapted layers when the backward graph exceeds device memory.

MusicGen's separate teacher-forced graph, delayed-codebook handling and valid-token
cross-entropy are implemented in `src/mg/train.h`. Ordinary generation training
is the default; prefix-masked training remains an optional ablation. See
[MUSICGEN_LORA.md](MUSICGEN_LORA.md) and
[GGML 0.27 training validation](TRAINING_GGML_027_VALIDATION.md).

See [TRAINING_CONTROLS.md](TRAINING_CONTROLS.md) for shared optimizer controls, broader cache coverage, random crops, FFN targets, and exact resume.
