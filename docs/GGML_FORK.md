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
