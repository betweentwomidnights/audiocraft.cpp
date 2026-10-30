# Native training validation on GGML 0.27

Validated 2026-10-29 on Windows 11 / RTX 5070 Laptop (driver 616.56), after merging
`master` through `96fa0a6` into `feature/native-lora-training` at `b75bf35`.
GGML is exactly `b77ae790e628270af46a8af1eca38441c23f1252`. No training implementation
changes or relaxed parity tolerances were needed for the upgrade.

## Build and integration results

| Check | CUDA build | Vulkan build |
| --- | --- | --- |
| Release build, all tools/tests | PASS | PASS |
| CTest (including CPU trainer fixtures) | 22/22 | 22/22 |
| MusicGen integration on selected GPU | PASS | PASS |
| MelodyFlow integration on selected GPU | PASS | PASS |
| Full-model F16 DoRA updates | PASS | PASS |
| Full-model adapter/state after exact resume | byte-identical | byte-identical |

GPU integration uses `AC_TRAIN_TEST_DEVICE=gpu`; Vulkan selects the NVIDIA device
with `AC_GPU=nvidia`. Tests cover warmup, accumulation, fresh crops, FFN targets,
AdamW settings, fixed evaluation, prompt sidecars, and rejected configurations.
The new Vulkan cross-entropy forward/backward implementation removes the previous
MusicGen training limitation on GGML 0.17.

Full-model smoke tests use official MusicGen-small and MelodyFlow F16 bases with
one training and one disjoint validation Koan cache entry. Each trains rank-32
DoRA in the final layer's attention and FFN projections for two optimizer updates,
with two microbatches/update, two-step warmup, fresh crops, and a checkpoint after
update one. MusicGen uses 100 frames (2 seconds); MelodyFlow uses 50 frames
(2 seconds) and ordinary logit-normal flow matching. Continuing the checkpoint to
update two produces the exact same adapter and optimizer/RNG state as the
uninterrupted run on each backend. This is a correctness smoke test, not a new
quality or long-window training claim.

## Independent AudioCraft / PyTorch checks

The existing opt-in scripts instantiate actual AudioCraft models from the intact
Gary installer environment (`torch 2.7.1+cu128`). PyTorch runs on CPU F32; F16 base
weights are rounded before constructing the reference. Native probes run on the
selected backend. Both additive LoRA and fully differentiated DoRA rows pass:

| Native configuration | MusicGen max gradient relative L2 | MelodyFlow max gradient relative L2 |
| --- | ---: | ---: |
| CPU, F32 base | 4.78e-7 | 2.06e-6 |
| CUDA, F16 base | 4.66e-7 | 1.26e-6 |
| CUDA, F32 base | 5.37e-7 | 2.16e-6 |
| NVIDIA Vulkan, F32 base | 4.79e-7 | 3.02e-6 |
| NVIDIA Vulkan, F16 base | strict loss check fails; see below | 1.46e-6 |

Eighteen passing model/backend/adapter configurations cover 164 conditions and 492 optimizer
updates. Conditions include zero/nonzero B, conditional, padded and null text;
MusicGen also checks full-sequence/prefix masks, while MelodyFlow checks ordinary
flow and edit-bridge/null-bridge targets. All losses, every adapter gradient and
three Adam updates are compared. Worst Adam parameter absolute difference is
2.83e-6 (existing bound 3e-5); maximum gradient relative L2 is 3.02e-6 (existing
bound 0.003). MusicGen additionally compares training logits with cached serving,
checks unchanged prefix codes, and requires exact base output at adapter strength
zero. No tolerance was changed.

### Precision boundaries

The additional F16 checks do **not** all pass the strict F32-reference oracle.
CPU/F16 fails the first loss check for both models/adapter types; first-loss
differences are about 1.56e-4 for MusicGen and 9.86e-4 for MelodyFlow, above the
unchanged 2e-5 bound. Replaying against GGML 0.17 produces exactly the same
losses, so this is a pre-existing precision boundary, not a new 0.27 regression.
Use an F32 base for strict CPU training parity.

MusicGen/F16 on NVIDIA Vulkan also misses that loss gate (about 6.06e-5 in the
first fixture). Its full-model updates and exact-resume smoke test pass, but
that does not establish strict oracle parity. Use F32 for Vulkan MusicGen
training when reference parity is required. The old Vulkan pin lacked the
cross-entropy training path, so an equivalent old-pin loss replay is unavailable.
MelodyFlow/F16 passes the Vulkan oracle. These observations are not resolved by
loosening the tests or reported as passing configurations.

Prepare caches against the same converted base file used for training. In
particular, rebuild MusicGen projected text caches when changing base precision;
matching model names alone do not prove identical conditioner projections.

Example replay (use `mf` for MelodyFlow and the corresponding probe):

```powershell
$env:AC_GPU='nvidia'
C:/dev/gary-localhost-installer/services/melodyflow/env/Scripts/python.exe `
  tests/mg_train_torch_parity.py `
  --runtime C:/dev/gary-localhost-installer/services/melodyflow `
  --gguf-path C:/dev/audiocraft.cpp/.venv/Lib/site-packages `
  --probe BUILD/bin/Release/mg_train_parity_probe.exe `
  --out OUTPUT --device vulkan --weight-type f32 --adapter-type dora-rows
```

## Evidence and limits

Local logs and replay commands are retained under `C:/ac7/`: `training-*-build.log`,
`training-*-ctest.log`, `training-*-*-integration.log`,
`training-oracles/summary.json` and per-case `results.json`, plus
`full-model-training/summary.json` and `resume-summary.json`.
Additional dtype checks and the old-pin CPU/F16 replay are in
`training-oracles-extra` and `training-oracles-old-half` respectively.
The executable build trees reuse the validated GGML 0.27 candidate builds; their
application checkout was explicitly advanced to the merged training commit.

Metal training remains unvalidated. Activation checkpointing, streaming cache
loading, and long-window/high-rank memory limits are not addressed here. The
inference precision fix from PR #7 remains applied in serving; training retains
its explicit differentiable operations and F32 matmul precision.
