# audiocraft.cpp

MusicGen and MelodyFlow in C++ on ggml — no PyTorch, no Python at inference time.

This is the sibling of [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp), built on
the same [`betweentwomidnights/ggml`](https://github.com/betweentwomidnights/ggml) fork at
the same pin. `sa3.cpp` covers stable-audio-3, Stable Audio Open and Foundation-1 for
[gary4local](https://github.com/betweentwomidnights/gary-localhost-installer); this repo
covers the two services still on PyTorch there — **gary** (MusicGen
`generate_continuation`) and **terry** (MelodyFlow `edit`).

Scope is deliberately narrow: only what those two services actually call. See
[docs/ROADMAP.md](docs/ROADMAP.md).

> **status: phase 6.** Both models run end to end and both match the PyTorch reference.
> `mf-edit` reproduces terry's settings at 0.9999179 on 30 s of audio — closer to
> torch-on-CPU than torch-on-GPU is. `mg-generate` is gary's whole transform, wav in and wav
> out, and reproduces torch **token for token from a raw 44.1 kHz file**: 6000/6000 codes
> over a 30 s continuation, with the decoded audio at cossim 1.0000000. It is also faster
> than torch on both CPU and GPU. `melodyflow-server` and `musicgen-server` serve both on the routes
> gary4juce already speaks, so gary4local's Tauri side only changes an `entryPoint` — leaving
> model publication and quantized tiers as the last of it.

## Build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/audiocraft.cpp.git
cd audiocraft.cpp

./build.sh cpu          # or: cuda | vulkan | metal | all   (windows: build.cmd cuda)
```

Needs CMake and a C++17 compiler (Visual Studio 2022 on Windows). CUDA needs the CUDA
Toolkit; Vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home); Metal is
macOS-only.

Converters and the parity harness need a small Python environment:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # posix: .venv/bin/python
```

The reference dumpers additionally need torch, transformers and audiocraft. Rather than
installing torch here, point them at one of gary4local's existing service venvs — see
[docs/T5.md](docs/T5.md).

## Try it

```bash
T5=~/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
.venv/Scripts/python.exe tools/convert_t5.py \
  --src "$T5/model.safetensors" --config "$T5/config.json" --tokenizer "$T5/tokenizer.json" \
  --out models/t5-base-encoder-0.1B-v1.0-F16.gguf --out-type f16

build/bin/Release/ac-tokenize --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf \
  --prompt "80s pop track with bassy drums and synth"

build/bin/Release/ac-textenc --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf \
  --prompt "80s pop track with bassy drums and synth" --out cppout/t5_hidden.f32
```

With the MelodyFlow ggufs built too, `mf-edit` is terry's whole transform, and its defaults
are the service's euler settings (25 steps, regularized 2/1, `lambda_kl` 0.2, guidance 4.0);
only the per-preset `--flowstep` varies:

```bash
build/bin/Release/mf-edit \
  --dit models/melodyflow-dit-t24-1.0B-v1.0-F32.gguf \
  --codec models/melodyflow-vae-48khz-v1.0-F32.gguf \
  --t5 models/t5-base-encoder-0.1B-v1.0-F32.gguf \
  --input in.wav --prompt "a dubby reggae bassline" --out out.wav
```

`mg-generate` is gary's whole transform — continue a track for 30 s, conditioned on a
description. `--greedy` is the mode that reproduces torch exactly; leave it off for the
sampling gary actually uses:

```bash
build/bin/Release/mg-generate \
  --lm models/musicgen-vanya-dnb-0.4B-v1.0-F16.gguf \
  --t5 models/t5-base-encoder-0.1B-v1.0-F16.gguf \
  --codec models/encodec-32khz-v1.0-F32.gguf \
  --input in.wav --prompt "drum and bass" --duration 30 --out out.wav
```

See [docs/MELODYFLOW_EDIT.md](docs/MELODYFLOW_EDIT.md),
[docs/MUSICGEN_LM.md](docs/MUSICGEN_LM.md) and
[docs/MUSICGEN_ENCODEC.md](docs/MUSICGEN_ENCODEC.md) for the parity numbers and what each
stage does, and [docs/SERVICES.md](docs/SERVICES.md) for the routes.

## Configuration

Backend and paths come from environment variables, so a downstream app sets them in the
process it spawns and never touches the CLI. `source ./env.sh` (Windows: `env.cmd`, or
`. .\env.ps1`) puts the built tools on `PATH`.

| | env var |
|---|---|
| base ggufs | `AC_MODELS_DIR` |
| device / gpu selection / cpu threads | `AC_DEVICE` `AC_GPU` `AC_THREADS` |
| flash attention | `AC_FLASH_ATTN` |

## Layout

```
src/            header-driven, like sa3.cpp
  ac/           shared across both models (T5, tokenizer, transformer, SEANet, LSTM)
  mf/           MelodyFlow: DiT, VAE, flow solver, pipeline
  mg/           MusicGen: LM, delay pattern, KV cache, EnCodec, pipeline
  serve/        base64, JSON, sessions -- the plumbing under the two HTTP services
tools/          CLI entry points, servers, converters, reference dumpers, cossim harness
vendor/         cpp-httplib and yyjson, for the servers only
tests/          CTest binaries + python converter tests
docs/           per-model porting notes, measured parity, ggml pin policy
```

`src/{gguf_model,nn,wav,audio_post,encoding,rng}.h` and `src/ac/{t5,tokenizer}.h` are ports
of the corresponding `sa3.cpp` files with the namespace changed (`sa3` → `ac`) and the env
prefix changed (`SA3_` → `AC_`); each carries a provenance line at the top.

## Testing

```bash
cd build && ctest -C Release --output-on-failure
```

Parity against PyTorch is the bar. `tools/dump_*_refs.py` writes reference activations,
the C++ tools write the same tensors as raw f32 in ggml memory order, and
`tools/cossim.py` compares them at a 0.9999 gate, and reports rms-envelope and
log-magnitude-spectrum cosines alongside it for audio, which is the fair measure for
precision tiers. [docs/T5.md](docs/T5.md), [docs/MELODYFLOW_VAE.md](docs/MELODYFLOW_VAE.md),
[docs/MELODYFLOW_DIT.md](docs/MELODYFLOW_DIT.md), [docs/MELODYFLOW_EDIT.md](docs/MELODYFLOW_EDIT.md),
[docs/MUSICGEN_LM.md](docs/MUSICGEN_LM.md) and
[docs/MUSICGEN_ENCODEC.md](docs/MUSICGEN_ENCODEC.md) are the worked examples, and between
them record six findings worth knowing before porting any audio model: `ggml_conv_1d` rounds
activations to F16, `ggml_gelu` is the tanh approximation, a stored rotary table may not
equal its closed form, an "optional" attention sink may be what keeps a null branch finite, a
graph executed more than once must have every input re-uploaded before every execution, and
the resampler in front of the model can cost more accuracy than the model does.

## Credits

[facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) is the
reference implementation of both models.
[sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp) is where the porting method,
the ggml fork, and most of the shared code come from.
[PABannier/encodec.cpp](https://github.com/PABannier/encodec.cpp) and
[ayutaz/vokra](https://github.com/ayutaz/vokra) were surveyed as prior art. Neither is used:
MusicGen's codec turned out to be HuggingFace's `EncodecModel` rather than Meta's original,
and `ac/seanet.h` already was that architecture — see
[docs/MUSICGEN_ENCODEC.md](docs/MUSICGEN_ENCODEC.md).

Model weights are the authors': MusicGen and MelodyFlow checkpoints are CC-BY-NC-4.0,
t5-base is Apache-2.0. This code is MIT.
