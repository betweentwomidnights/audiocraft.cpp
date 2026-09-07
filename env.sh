# source ./env.sh
AC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
for d in build-cuda build-vulkan build-metal build-all build; do
  if [ -d "$AC_ROOT/$d/bin/Release" ]; then export PATH="$AC_ROOT/$d/bin/Release:$PATH"; break
  elif [ -d "$AC_ROOT/$d/bin" ]; then export PATH="$AC_ROOT/$d/bin:$PATH"; break; fi
done
export AC_MODELS_DIR="$AC_ROOT/models"
echo "[ac] environment ready (AC_MODELS_DIR=$AC_MODELS_DIR)"
