Audiocraft native runtime

This folder includes both servers, training/inference tools, CPU/GPU backends,
the Visual C++ runtime and the CUDA runtime. Model weights are separate.

Inspect the installation without models:
  musicgen-server --version
  musicgen-server --props
  melodyflow-server --props

Start one or both services with converted GGUF models:
  musicgen-server --models-dir C:\path\to\models
  melodyflow-server --models-dir C:\path\to\models

Default addresses: http://127.0.0.1:8000 and http://127.0.0.1:8002.
AC_MODELS_DIR, AC_DEVICE, AC_GPU, AC_HOST and per-service port variables are
documented in RUNTIME_README.md. Use --help for tool options. Training takes
prepared GGUF caches; it does not require a Python model in this folder.
