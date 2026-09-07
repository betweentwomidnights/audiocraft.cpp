# MelodyFlow's SEANet VAE

MelodyFlow's compression model is a quantizer-free SEANet autoencoder: 48 kHz **stereo**,
`ratios [8, 8, 6, 5]` so the hop is 1920 samples and the latent runs at **25 Hz**, snake
activations, weight-normalised convolutions, reflect padding, and a 2-layer LSTM in the
bottleneck. The encoder emits `mean||scale` (256 channels) and the decoder consumes 128.

It is the same network as MusicGen's EnCodec 32 kHz — different ratios, ELU instead of
snake, mono instead of stereo, and an RVQ where this one has nothing — so `src/ac/seanet.h`
is written once and driven entirely from GGUF metadata. Phase 5 adds the RVQ.

terry's window is fixed: `encode_within_latent_window` crops to `750 latents x 1920
samples` = exactly 30 s, at 48 kHz stereo, and pitch-correctness depends on that rate
(`localhost_melodyflow.py:49-57` records what went wrong when an earlier version resampled
to 32 kHz instead).

## Conversion

```bash
MF=~/.cache/huggingface/hub/models--facebook--melodyflow-t24-30secs/snapshots/*/
.venv/Scripts/python.exe tools/convert_seanet.py \
  --src "$MF/compression_state_dict.bin" \
  --out models/melodyflow-vae-48khz-v1.0-F32.gguf
```

120 tensors, 59,660,354 parameters, F32. The converter reads the raw audiocraft pickle
(`tools/audiocraft_ckpt.py`, `weights_only=True` with audiocraft's own OmegaConf
allowlist) and derives every shape from the embedded `xp.cfg`, so it will convert
MusicGen's codec unchanged once the RVQ path lands.

Everything constant is folded at conversion time, exactly as `sa3.cpp` does for Oobleck:

| folded | why |
|---|---|
| `weight_norm` → `g * v / ‖v‖` | `MelodyFlow.get_pretrained` does the same with `remove_parametrizations` before inference |
| snake's `1 / (alpha + 1e-9)` | a per-channel constant the graph would otherwise recompute every call |
| ConvTranspose1d → col2im layout | `[in, out, k]` → ggml `[in, k*out]`, so the graph is a matmul plus `ggml_col2im_1d` |
| `b_ih + b_hh` | `nn.LSTM` adds both unconditionally on every step |

The converter fails on anything it was not written for — a gated activation, a causal
SEANet, `true_skip=False`, a decoder `final_activation`, an unrecognised quantizer — and
raises if any source tensor goes unconsumed, so a checkpoint with a different topology
cannot convert into a silently wrong codec.

## Two findings worth carrying forward

### `ggml_conv_1d` rounds activations to F16

This cost the encoder 0.0016 of cosine similarity, 30x our parity gate, and it is invisible
from the call site. `ggml_conv_1d` builds its im2col matrix with `dst_type = GGML_TYPE_F16`
for every non-BF16 kernel (`ggml/src/ggml.c`), and that matrix is the **activations**, not
the weights — so an F32 model on an F32 backend still loses precision, once per
convolution. SEANet stacks sixteen of them.

`nn::conv_1d_f32` is the identical computation with an F32 im2col. Switching to it moved
the encoder latent from `cossim 0.9997439 / max abs err 8.2e-01` to `cossim 1.0000000 /
3.2e-05`, and the decoded audio from `3.8e-04` to `8.3e-07`.

The cost is memory: the im2col buffer is `out_frames * in_channels * kernel` floats, and
the widest layer of a 30 s encode is 1.44M frames x 64 channels x kernel 3 = 1.1 GB. Peak
working set for a 30 s window is 2.5 GB encoding and 3.8 GB decoding. That is fine on a
desktop CPU and uncomfortable on an 8 GB GPU; the fix when it matters is to tile the
convolutional stack over time the way `sa3.cpp` tiles its decoder (`plan_chunks`,
`--chunked-decode`), which is exact for a finite-receptive-field stack. Not needed yet.

Note `sa3.cpp`'s `sat/oobleck.h` calls `ggml_conv_1d` directly and so carries the same
rounding. It passes its own gate, but the same substitution is available if its numbers
ever matter more.

**The same error comes back on CUDA, from a different direction.** ggml creates its cuBLAS
handle with `CUBLAS_TF32_TENSOR_OP_MATH`, so an F32 GEMM computes at ten mantissa bits and
the encoder lands at cossim 0.9998489 / 8.2e-01 again — the same magnitude, for the same
reason. Neither `ggml_mul_mat_set_prec` nor `GGML_CUDA_CUBLAS_COMPUTE_TYPE` can turn it off.
Details, measurements and the one-line patch are in
[GGML_FORK.md](GGML_FORK.md#open-every-f32-gemm-on-cuda-silently-runs-at-tf32); with it, the
encoder returns to cossim 1.0000000 for about 10% more time.

### The bottleneck LSTM belongs in the graph

ggml has no LSTM operation and the recurrence is genuinely sequential, so the obvious move
was to run it on the host between two half-graphs — the staging `sa3.cpp` already uses for
T5 → DiT → autoencoder, and for its numeric conditioners. That was implemented first and
measured, on a 30 s window (1024 hidden, 2 layers, 750 steps), CPU:

| | encode | decode |
|---|---:|---:|
| host LSTM, scalar GEMV | 9.65 s (conv 2.87 + **lstm 6.75**) | 10.15 s (**lstm 6.68** + stack 3.45) |
| unrolled into the graph | **2.57 s** | **3.10 s** |

The host loop spent more than twice as long on the bottleneck as on the entire
convolutional stack around it, at roughly 1.85 GFLOP/s — scalar speed. `ac/lstm.h` unrolls
the recurrence instead: about 17 nodes per step per layer, so ~26k nodes for a 30 s window,
and every matmul goes through ggml's threaded, vectorised, backend-portable path. It also
keeps the codec as one graph on any backend rather than needing a host round trip in the
middle, which matters once terry runs on a GPU.

Two details make the unroll cheap rather than quadratic:

- **The input projection is hoisted.** `W_ih @ X` is not recurrent, so it is one matmul
  over all timesteps per layer — half the arithmetic, out of the loop.
- **The per-step write is in place.** `ggml_set_1d` copies the whole `[hidden, frames]`
  buffer per step, which is quadratic in sequence length; `ggml_set_1d_inplace` writes only
  its own slice, and the chain stays correctly ordered by its own data dependency.

## Measured parity

Reference: `tools/dump_mf_vae_refs.py`, which loads the real compression model through
`audiocraft.models.loaders.load_compression_model` and strips weight-norm exactly as
`MelodyFlow.get_pretrained` does. Both sides read the *same* 48 kHz 16-bit wav the dumper
writes, so the comparison measures the codec and not two different resamplers.

Encode and decode are compared separately and both are deterministic. `z` is the posterior
**mean**, not a `vae_sample` draw: matching torch's RNG stream is a different problem from
matching the codec, and feeding both sides the same `z` keeps the two halves independently
falsifiable.

30 s stereo, F32 GGUF, CPU:

| checkpoint | CPU cossim | CPU max abs err | CUDA cossim |
|---|---:|---:|---:|
| `mf_vae_latent` — encoder `mean‖scale` [256, 750] | 1.0000000 | 3.2e-05 | 0.9998489 |
| `mf_vae_audio` — decoder output [2, 1440000] | 1.0000000 | 2.0e-05 | 0.9999998 |

The CUDA encoder figure is the TF32 issue above, not a graph difference; with TF32 disabled
it is 1.0000000. Encode takes 0.40 s and decode 0.47 s on an RTX 5070 Laptop, against
2.57 s and 3.10 s on this CPU.

Intermediate taps at 10 s, against the torch per-layer dump:

| tap | vs | cossim | max abs err |
|---|---|---:|---:|
| `mf_vae_enc_pre` (bottleneck in) | `encoder.model[12]` | 1.000000000 | 1.9e-05 |
| `mf_vae_enc_lstm` (bottleneck out) | `encoder.model[13]` | 1.000000000 | 6.4e-05 |

`tests/lstm_test.cpp` pins the LSTM independently against a torch fixture on every backend
the build hosts: max abs err 6.0e-08, i.e. float32 round-off.

A full round trip (`--roundtrip`, which does sample the posterior) reconstructs the input at
RMS-envelope cosine 0.999616 and log-magnitude-spectrum cosine 0.958941. That gap is the
VAE's own lossy reconstruction through a 25 Hz, 128-dimensional latent plus the posterior
noise draw, not an implementation error — the deterministic comparisons above are what
tests the implementation.

## Reproducing

```bash
# reference (needs torch, torchaudio and audiocraft -- use terry's venv)
MFDIR=/c/dev/gary-localhost-installer/services/melodyflow
PYTHONPATH="C:\\dev\\gary-localhost-installer\\services\\melodyflow" \
  $MFDIR/env/Scripts/python.exe tools/dump_mf_vae_refs.py \
    --audio testdata/music.wav --out refdata30 --seconds 30

# ours
build/bin/Release/ac-vae --codec models/melodyflow-vae-48khz-v1.0-F32.gguf \
  --encode refdata30/input48k.wav --out-latent cppout30/mf_vae_latent.f32
build/bin/Release/ac-vae --codec models/melodyflow-vae-48khz-v1.0-F32.gguf \
  --decode refdata30/mf_vae_z.f32 --out cppout30/mf_vae_audio.f32

.venv/Scripts/python.exe tools/cossim.py --ref refdata30 --cpp cppout30
```

Add `--taps` to both sides to walk a divergence layer by layer: the dumper writes
`mf_vae_enc<N>.npy` / `mf_vae_dec<N>.npy` for every top-level module, and `ac-vae` writes
the bottleneck tensors under the same names `tools/cossim.py` knows.

`--roundtrip in.wav --seed N --out out.wav` runs the whole thing including `vae_sample`,
for listening rather than for parity.

`tests/gen_lstm_fixture.py` regenerates the fixture arrays embedded in `tests/lstm_test.cpp`
if the cell implementation ever needs re-pinning.
