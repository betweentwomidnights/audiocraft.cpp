# audiocraft.cpp

This is part of a project to make
[gary-localhost-installer](https://github.com/betweentwomidnights/gary-localhost-installer)
truly cross-platform and contain zero bloated python dependencies.

We expose only the stuff gary4juce requires of these two audiocraft models. audiocraft is
really old at this point, but I don't believe in deprecating models. As long as they produce
interesting samples based on our human input audio, I am going to continue supporting them
inside gary.

MusicGen `generate_continuation` and MelodyFlow `edit`, in C++ on
[ggml](https://github.com/betweentwomidnights/ggml). No PyTorch, no Python at inference time.
Sibling of [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp), same ggml fork, same pin.

Both match the PyTorch reference. `mg-generate` reproduces gary's whole transform token for
token from a raw 44.1 kHz file — 6000/6000 codes over a 30 s continuation — and `mf-edit`
reproduces terry's settings at cosine similarity 0.9999179 on 30 s of audio.

## Build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/audiocraft.cpp.git
cd audiocraft.cpp

./build.sh cpu          # or: cuda | vulkan | metal | all   (windows: build.cmd cuda)
```

Needs CMake and a C++17 compiler (Visual Studio 2022 on Windows). CUDA needs the CUDA
Toolkit; Vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home); Metal is
macOS-only.

## Run

```bash
# gary: continue 30 s from the tail of a wav
mg-generate --lm models/musicgen-*.gguf --t5 models/t5-base-encoder-*.gguf \
            --codec models/encodec-32khz-*.gguf \
            --input in.wav --duration 30 --out out.wav

# terry: transform a wav with one of the 33 presets
mf-edit --dit models/melodyflow-dit-*.gguf --t5 models/t5-base-encoder-*.gguf \
        --codec models/melodyflow-vae-*.gguf \
        --input in.wav --variation "8bit" --out out.wav
```

`melodyflow-server` and `musicgen-server` serve the same routes gary4juce already speaks, so
the plugin talks to them unmodified.

## Docs

Per-model write-ups — architecture, the parity numbers, and what is deliberately left out —
live in [docs/](docs/): [MUSICGEN_LM.md](docs/MUSICGEN_LM.md),
[MUSICGEN_ENCODEC.md](docs/MUSICGEN_ENCODEC.md), [MELODYFLOW_EDIT.md](docs/MELODYFLOW_EDIT.md),
[SERVICES.md](docs/SERVICES.md), [ROADMAP.md](docs/ROADMAP.md).

## Credits

[audiocraft](https://github.com/facebookresearch/audiocraft) for MusicGen and MelodyFlow,
[ggml](https://github.com/ggml-org/ggml) for the tensor library. Model weights keep their
own licences — MusicGen and MelodyFlow are CC-BY-NC.
