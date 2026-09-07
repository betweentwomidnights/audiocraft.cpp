#!/usr/bin/env bash
# Build audiocraft.cpp for one backend into its own build dir (backends coexist for A/B).
#
# Usage: ./build.sh [cpu|cuda|vulkan|metal|all]   (default: cpu)
set -euo pipefail

BACKEND="${1:-cpu}"
EXTRA=("${@:2}")

case "$BACKEND" in
  cpu)    DIR=build         ; FLAGS=(-DGGML_CUDA=OFF) ;;
  cuda)   DIR=build-cuda    ; FLAGS=(-DAC_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native) ;;
  vulkan) DIR=build-vulkan  ; FLAGS=(-DAC_VULKAN=ON) ;;
  metal)  DIR=build-metal   ; FLAGS=(-DAC_METAL=ON) ;;
  all)    DIR=build-all     ; FLAGS=(-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DAC_CUDA=ON -DAC_VULKAN=ON) ;;
  *) echo "unknown backend: $BACKEND  (cpu|cuda|vulkan|metal|all)" >&2; exit 1 ;;
esac

echo "[ac] configuring $BACKEND -> $DIR/"
cmake -S . -B "$DIR" -DCMAKE_BUILD_TYPE=Release "${FLAGS[@]}" "${EXTRA[@]}"
cmake --build "$DIR" --config Release --parallel
echo "[ac] built $BACKEND -> $DIR/bin/"
