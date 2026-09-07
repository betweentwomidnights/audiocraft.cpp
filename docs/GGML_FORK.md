# The ggml submodule

`audiocraft.cpp` pins an exact commit of
[`betweentwomidnights/ggml`](https://github.com/betweentwomidnights/ggml) — **the same
fork and the same pin `sa3.cpp` uses**, so the two repos stay trivially comparable and a
backend fix validated in one is the identical code in the other.

Current pin: `19c5421c9314893db83ab93575d34b9cb828516f`
(`git describe` → `sa3-training-v1-metal-100-g19c5421c`, upstream base ggml `v0.17.0`).

The fork's patch stack — CPU/CUDA/Vulkan/Metal autodiff and backend work, the Q4_K_M
`get_rows` fix, the wide-row `SET` fix, quantized-`src0` `OUT_PROD` — is documented in
**`sa3.cpp/docs/GGML_FORK.md`**, which is the source of truth. None of it is inference
forward-op work; the models here use stock ggml operations.

Ops this repo depends on that are worth naming, because they are the ones a future
upstream bump could disturb:

| op | used by |
|---|---|
| `ggml_pad_reflect_1d` | SEANet's `pad_mode: reflect` (both codecs) |
| `ggml_col2im_1d` | SEANet decoder ConvTranspose1d, via the GEMM+col2im formulation |
| `ggml_conv_transpose_1d` | ditto (alternate path) |
| `ggml_elu` | EnCodec 32 kHz activation |
| `ggml_rope_ext` (NeoX) | MelodyFlow DiT positional embedding |
| `ggml_flash_attn_ext` | optional attention path (`AC_FLASH_ATTN`) |

## Open: every F32 GEMM on CUDA silently runs at TF32

`ggml_backend_cuda_context::cublas_handle` creates its cuBLAS handle with

```cpp
CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
```

(`ggml/src/ggml-cuda/common.cuh`). TF32 keeps 10 mantissa bits -- about half precision --
so on Ampere and later *every* F32 matmul that reaches cuBLAS is computed at roughly F16
accuracy, whatever the tensors say.

This is upstream ggml, not a fork patch: `git log -L` dates the line to the 2024-03-27
"sync : adapt to CUDA changes" commit. It is presumably a deliberate trade for language
models. It is not a good trade for a deep convolutional audio codec.

**There is no way to opt out from the calling side.** `ggml_mul_mat_set_prec(GGML_PREC_F32)`
and the `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` environment variable both only select
`CUBLAS_COMPUTE_32F`, which the handle's math mode then downgrades anyway. Measured: both
leave the output bit-identical.

Measured on an RTX 5070 Laptop against float32 torch, with a one-line local patch gating the
math mode on an environment variable:

| | TF32 on (today) | TF32 off | cost |
|---|---|---|---|
| MelodyFlow VAE encoder, 30 s | cossim 0.9998489, err 8.2e-01 | **1.0000000**, err 2.9e-05 | 0.408 s -> 0.448 s (+10%) |
| MelodyFlow DiT F32, 750 frames | 0.9999958 | 0.9999975 | 0.227 s -> 0.305 s (+34%) |
| MelodyFlow DiT F16, 750 frames | 0.9999286 | **0.9999768** | 0.225 s -> 0.228 s (free) |

The VAE encoder is the one that matters: with TF32 on it fails this repo's 0.9999 parity
gate, and the error is the same magnitude as the F16-im2col bug documented in
[MELODYFLOW_VAE.md](MELODYFLOW_VAE.md) -- sixteen stacked convolutions accumulate it. For the
DiT, turning TF32 off is free at F16 and buys a 3x smaller error.

The patch that produced those numbers, for whoever picks this up:

```cpp
// ggml/src/ggml-cuda/common.cuh, in cublas_handle(int device)
CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device],
    getenv("GGML_CUDA_TF32") && getenv("GGML_CUDA_TF32")[0] == '0'
        ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH));
```

It is deliberately **not** applied to the pinned submodule. Changing the shared fork means
following the pin policy below -- build and test every backend, push the branch, move the
gitlink -- and it would change `sa3.cpp`'s numerics too, so it is a decision for the fork
owner rather than something to slip in. It is also worth raising upstream: any ggml consumer
running convolutions or long accumulation chains in F32 on CUDA is affected and has no way
to know.

Until then, CUDA results in this repo are reported with TF32 on, which is what a fresh clone
produces.

## Pin policy

Same policy as `sa3.cpp`, and for the same reason:

- The gitlink in each `audiocraft.cpp` commit is the source of truth. There is
  deliberately **no `branch = ...`** line in `.gitmodules`, and nobody should run
  `git submodule update --remote`.
- Fork branches are development lines, not dependency selectors.
- Bump the pin only after building and running the registered CTest suite on every
  affected backend.

Read the pin from a checkout rather than trusting this file:

```bash
git -C ggml rev-parse HEAD
git -C ggml describe --tags
```

## Updating an existing clone

```bash
git -c fetch.recurseSubmodules=false pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

## A note on the default branch

At the time of writing, the fork's default branch tip is `a4a52c4a` (the merge of PR #5,
"Shared GGML 0.17 candidate for SA3 and ACE training"). Its **tree is byte-identical** to
the pinned `19c5421c` — `git diff 19c5421c a4a52c4a` is empty — so pinning the same commit
`sa3.cpp` does costs nothing and buys a directly comparable checkout.
