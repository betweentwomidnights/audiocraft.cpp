# Third-party components in native runtime packages

- GGML: MIT license, `LICENSE-ggml.txt`.
- cpp-httplib: MIT license, `LICENSE-cpp-httplib.txt`.
- yyjson: MIT license, `LICENSE-yyjson.txt`.
- Windows packages carry Microsoft Visual C++ redistributable runtime files
  from the matching Visual Studio toolset, under Microsoft's redistribution terms.
- The standalone Windows package carries NVIDIA CUDA runtime/cuBLAS DLLs and
  their redistribution terms in `NVIDIA-CUDA-EULA.txt`. Split GPU backend
  archives omit the CUDA runtime; the supervisor supplies its shared runtime.
- macOS packages use system frameworks. Model weights and their separate
  licenses are not included in any runtime archive.

The project license is included as `LICENSE`.
