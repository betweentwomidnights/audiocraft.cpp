# The ggml submodule

`audiocraft.cpp` pins an exact commit of
[`betweentwomidnights/ggml`](https://github.com/betweentwomidnights/ggml), the fork shared
with `sa3.cpp` and `acestep.cpp`.

Current pin: `07f9348a`, the shared Vulkan candidate
([betweentwomidnights/ggml#10](https://github.com/betweentwomidnights/ggml/pull/10)) that every
consumer of the fork pins. It sits on `fff93d27`, the previous tip of
`feature/shared-sa3-acestep-v0.17`. That tip is `19c5421c` (upstream ggml `v0.17.0` plus the
fork's patch stack) with the two CUDA commits from betweentwomidnights/ggml#6 on top: the TF32
opt-out below, and a contiguity check for the transposed copy (see
[MUSICGEN_LM.md](MUSICGEN_LM.md)). #10 merges three Vulkan fixes and adds tests:

- #7, `GGML_PREC_F32` keeps an F32 x F32 matmul's operands in fp32. Without it, Vulkan rounds
  them to fp16.
- #8, the batch stride of an in-place matmul src comes from `nb[2]`. This is the KV-cache bug
  `tests/kv_attention_test.cpp` pins.
- #9, a Vulkan `PAD_REFLECT_1D` kernel. Without it, SEANet produced noise on Vulkan.

On Vulkan, that brings the MelodyFlow VAE and EnCodec to cossim 1.0000000 against torch, and
greedy MusicGen to 100% of tokens.

The rest of the fork's patch stack — CPU/CUDA/Vulkan/Metal autodiff and backend work, the Q4_K_M
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

## CUDA: opting out of TF32 for F32 matmuls

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
math mode on an environment variable. Accuracy first:

| | TF32 on (today) | TF32 off |
|---|---|---|
| MelodyFlow VAE encoder, 30 s | cossim 0.9998489, err 8.2e-01 | **1.0000000**, err 2.9e-05 |
| MelodyFlow DiT F32, 750 frames | 0.9999958 | 0.9999975 |
| MelodyFlow DiT F16, 750 frames | 0.9999286 | **0.9999768** |

The VAE encoder is the one that matters: with TF32 on it fails this repo's 0.9999 parity
gate, and the error is the same magnitude as the F16-im2col bug documented in
[MELODYFLOW_VAE.md](MELODYFLOW_VAE.md) -- sixteen stacked convolutions accumulate it. For the
DiT, turning TF32 off buys a 3x smaller error at F16.

### What it costs

**Applying the patch costs nothing.** It is one `getenv` at cuBLAS handle creation, once per
process, and with `GGML_CUDA_TF32` unset the behaviour is bit-identical to today. Only
opting out has a price, and it is not where the single-sample numbers first suggested.

Counterbalanced ABBA, six samples each, medians -- following this file's own rule that a
laptop GPU throttles across a sequence, so a plain "run A then B" measures the ordering:

| | TF32 on | TF32 off | delta | spreads |
|---|---:|---:|---:|---|
| DiT F16, 750 frames | 0.227 s | 0.231 s | +1.5% | overlap -- not resolvable |
| DiT F32, 750 frames | 0.237 s | 0.299 s | +25.9% | disjoint |
| VAE encode, 30 s | 0.406 s | 0.414 s | +2.1% | overlap -- not resolvable |

Only the F32 DiT is a real slowdown, and that build is 3.9 GB and not the one that ships.
For the configuration terry actually runs -- F16 DiT, F32 VAE -- a full 125-forward edit
goes from about 28.4 s to about 28.8 s, roughly 1%, in exchange for the encoder going from
below the parity gate to exact. The F16 case is the interesting one: half-precision weights
are *already* the dominant error, so TF32 on top buys no speed and costs accuracy.

The patch that produced those numbers, for whoever picks this up:

```cpp
// ggml/src/ggml-cuda/common.cuh, in cublas_handle(int device)
CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device],
    getenv("GGML_CUDA_TF32") && getenv("GGML_CUDA_TF32")[0] == '0'
        ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH));
```

### How it is wired

The **fork default is unchanged**: with `GGML_CUDA_TF32` unset, the math mode is exactly
what it was, so `sa3.cpp` and `acestep.cpp` built against this branch produce bit-identical
output to before. Only a caller that asks moves.

`audiocraft.cpp` asks, for its own processes only. `ac::prefer_f32_matmuls_once()` in
`src/gguf_model.h` sets `GGML_CUDA_TF32=0` before the first backend is created, unless the
caller already set `GGML_CUDA_TF32` or passed `AC_CUDA_TF32=1`. So:

| | result |
|---|---|
| default | full F32 accumulation, matching CPU |
| `AC_CUDA_TF32=1` | ggml's default TF32, ~2% faster, 1.5e-4 less accurate |
| `GGML_CUDA_TF32=` anything | wins over both; the escape hatch for A/B testing |

### Still worth raising upstream

Any ggml consumer running convolutions or long F32 accumulation chains on CUDA is affected
and has no way to know: the tensors say F32, the backend reports F32, and the arithmetic is
not. A `ggml_backend_cuda_set_tf32()` or a `supports_op`-visible flag would be a better
answer than an environment variable; the variable is what fits in one line without adding
public API surface to a fork.

### Before this pin is published

Per the pin policy below, and because the branch changes CUDA numerics for everyone who
takes it:

1. Build `sa3.cpp` and `acestep.cpp` against this branch on CUDA.
2. Confirm their outputs are unchanged with `GGML_CUDA_TF32` unset — that is the whole point
   of leaving the default alone, and it is the regression test.
3. Optionally measure what they gain from `GGML_CUDA_TF32=0`; both have convolutional
   decoders, so `sa3.cpp`'s Oobleck is a plausible beneficiary.
4. Then push the branch and record the pin here.

## Vulkan: strided batches in mul_mat

Backport of upstream
[ggml-org/llama.cpp#28956](https://github.com/ggml-org/llama.cpp/pull/28956). The Vulkan
backend read a row-contiguous view with strided batches in place, but passed a packed batch
stride (`ne00*ne01`). The KV cache's K history is exactly that view, so on Vulkan every
attention head past the first read the wrong keys. `kv_attention` failed on Vulkan, and
greedy MusicGen matched torch on 0.02% of tokens. With the fix it matches on 100%. The
details are in [MUSICGEN_LM.md](MUSICGEN_LM.md); the fork-side numbers and
`test-backend-ops` cases are in betweentwomidnights/ggml#8.

It changes no result that was right before: a contiguous tensor gets the same stride as
before, and only views like this one move.

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

## Where this branch sits

```
19c5421c   sa3.cpp's pin, upstream v0.17.0 + the fork's patch stack
  |
  +-- de8870f4   cuda : allow opting out of TF32 for F32 matmuls
  +-- fff93d27   cuda : require a contiguous destination for the transposed copy
  |              (tip of feature/shared-sa3-acestep-v0.17, via #6)
  |
  +-- 217f0f2d   vulkan : honor GGML_PREC_F32 for f32 x f32 mul_mat (#7)
  +-- d6e6604f   vulkan : read the batch stride of an in place mul_mat src from nb[2] (#8)
  +-- 5cb55640   tests : SEANet-shaped PAD_REFLECT_1D cases (#9, on d12e8055)
  +-- 07f9348a   tests : MUL_MAT cases with GGML_PREC_F32
                 (shared Vulkan candidate, #10)   <- audiocraft.cpp pins this
```

Read the pin from a checkout rather than trusting this file:

```bash
git -C ggml rev-parse HEAD          # 07f9348a...
git -C ggml log --oneline 19c5421c..HEAD
```
