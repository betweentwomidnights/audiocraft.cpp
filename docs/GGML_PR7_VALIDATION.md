# GGML 0.27 and MelodyFlow precision

PR #7 advances the shared GGML pin from
`60f49e09ce67df544831a357b654318ec2722fb0` to
`b77ae790e628270af46a8af1eca38441c23f1252`.

The original cross-pin validation found a pre-existing Vulkan MelodyFlow edit
accuracy failure at both pins. F32 DiT matmuls were using the backend's default
precision, which permits F16 operand staging. A small per-forward difference then
accumulated through the inversion/generation solve.

`dit_set_matmul_precision` explicitly requests `GGML_PREC_F32` for matmuls with
two F32 operands, before graph allocation. Both `DitRunner` (CLI and HTTP editing)
and `ac-dit` apply it. F16-weight products retain their original precision policy.
The graph-level `dit_precision` test checks both cases without needing model files.

On the RTX 5070 Laptop, the old pin's 30-second replay against the saved AudioCraft
reference changes as follows with this application correction:

| Vulkan output | Before cosine | After cosine |
| --- | ---: | ---: |
| DiT velocity, 750 frames | 0.9999087 | 1.0000000 |
| Final edit latent | 0.9877598 | 0.9999892 |
| Decoded edit audio | 0.9851339 | 0.9999822 |

The existing cosine gate remains **0.9999**. Noise input, model files, prompt,
solver, regularization and guidance settings remain fixed. CUDA also passes the
same DiT/edit reference checks with the precision correction on the old pin.

CUDA uses audiocraft's existing default `GGML_CUDA_TF32=0`; opting into TF32 is a
different precision/performance choice. Explicit F32 operands can cost Vulkan
performance relative to implicit downcasting. The F16 model tier remains available
when that tradeoff is wanted.

Training is a separate validation surface. The LoRA branch has deliberately not
been integrated into this inference-only bump; its autodiff and optimizer tests
must be rerun against the new GGML pin before that branch merges.
