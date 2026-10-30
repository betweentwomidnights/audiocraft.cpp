# Audiocraft native runtime releases

One `audiocraft` bundle serves both `musicgen-server` and `melodyflow-server`,
analogous to SA3's shared SA3/SAT bundle. Model weights are not included.
The downstream package contract is
`gary-localhost-installer/docs/native-runtime-packages.md`.

## Package layout

| Archive | Contents |
| --- | --- |
| `audiocraft-vX.Y.Z-windows-x64-core.zip` | Both servers, native training/inference CLIs, ggml core and every CPU variant, matching app-local VC runtime, licenses, docs, BUILD-INFO.json |
| `audiocraft-vX.Y.Z-windows-x64-cuda.zip` | ggml-cuda.dll only |
| `audiocraft-vX.Y.Z-windows-x64-vulkan.zip` | ggml-vulkan.dll only |
| `audiocraft-vX.Y.Z-windows-x64-standalone.zip` | Core, both GPU backends, CUDA runtime DLLs and EULA |
| `audiocraft-vX.Y.Z-macos-universal.zip` | Universal tools and ggml dylibs, arm64 Metal/CPU and x86_64 CPU, licenses and build metadata |

Archives are flat. Gary4local installs core plus one GPU backend, and supplies
its existing shared CUDA 12.8 runtime on PATH. Direct Windows users take the
standalone archive. SHA256SUMS and SHA256SUMS-macos contain LF-terminated hashes.
BUILD-INFO.json records source/ggml commits, version, build flavor and provenance.
Each service installation keeps its own ggml libraries; this bundle does not
install DLLs globally. Python cache preparation helpers stay in the source tree;
native trainers accept prepared GGUF caches without a Python model runtime.

## Offline probes and configuration

Both servers implement `--version` (bare CMake version) and `--props` (JSON with
service, runtime, version and registered devices). These probes load no models,
bind no port and write no files. A successful GPU install must report the selected
GPU backend in `devices`; a CPU-only result does not prove a GPU overlay works.
The `service` values are `musicgen` and `melodyflow`, while the bundle identity is
`audiocraft`. Existing `/health` and application endpoints remain unchanged.

| Environment | Meaning |
| --- | --- |
| AC_MODELS_DIR | GGUF model directory, default `models` |
| AC_DEVICE | Requested backend/device, e.g. cpu, cuda or vulkan |
| AC_GPU | Index within the selected backend or description filter, e.g. nvidia |
| AC_THREADS | CPU graph thread count; CI's tiny trainer fixtures use one |
| AC_HOST | Listen address, default 127.0.0.1 |
| AC_PORT | Port for a single supervised process |
| AC_MUSICGEN_PORT / AC_MELODYFLOW_PORT | Per-service overrides of AC_PORT; defaults 8000/8002 |
| AC_ADAPTERS_DIR | Adapter catalog; default models/adapters |

CLI options override environment values. Training precision boundaries are
documented in TRAINING_GGML_027_VALIDATION.md; Metal training is not claimed by
the release packaging checks.

## Build and verify

The packaging scripts run the same way locally and in CI. Windows uses VS2022,
CUDA 12.8.1 and Vulkan SDK 1.4.350.0. Portable builds disable GGML_NATIVE, load
backends dynamically, build every CPU variant and use ggml's portable CUDA
architecture defaults. `-CudaArch native` is a local smoke override and requires
a separate build directory. Tests run on CPU (`AC_TEST_DEVICE=cpu`), and staged
executables are verified from an unrelated directory with the build/toolkit PATH
removed. A Windows loader check requires all four VC runtime DLLs to come from
the core folder rather than System32.

```powershell
./ci/package-windows.ps1 -Version v0.1.0 -CpuOnly -BuildDir build-dist-smoke -OutDir dist-smoke
./ci/package-windows.ps1 -Version v0.1.0
```

macOS builds on Apple Silicon/macOS 15 with Rosetta to test both slices. It embeds
Metal shaders, rewrites non-system dylib imports to @loader_path, signs binaries,
and submits the final zip for notarization when signing credentials are available.
Release builds require Developer ID and notarization credentials. Ad-hoc dry runs
are allowed and are not distributable signed releases.

## Workflow and release checklist

`release.yml` runs on published releases or manual dispatch. Empty `tag` is a dry
run: artifacts are retained for 14 days without publishing. A supplied tag rebuilds
and attaches to an existing release; version mismatches fail. Published packages
get GitHub provenance attestations. PR checks build a Windows CPU smoke package
and the universal macOS dry-run package; they do not prove portable Windows GPU
packaging. Full dispatch/release builds include both Windows GPU backends.

1. Review and merge packaging changes. Configure the sibling secret names:
   MACOS_CERT_P12 (base64), MACOS_CERT_PASSWORD, APPLE_TEAM_ID,
   APPLE_NOTARY_KEY_P8 (base64), APPLE_NOTARY_KEY_ID, APPLE_NOTARY_ISSUER_ID.
2. Dispatch with no tag, validate extracted Windows core+CUDA/Vulkan overlays on
   real GPUs, and verify universal macOS startup/signatures/notarization. The
   `--props` device list must prove the backend actually loaded.
3. Set the CMake version once, create its matching release tag, and publish only
   after the dry-run packages work. Model artifacts remain separate.
4. Review SHA-256 assets and attestations. A downstream Gary manifest migration
   is a separate change that pins asset URLs+hashes; this workflow changes no
   installed runtime or manifest. Do not rebuild/clobber assets already pinned
   downstream, since rebuilding changes package hashes.
