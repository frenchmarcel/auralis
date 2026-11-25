#!/usr/bin/env bash
# Build the Auralis CUDA core into auralis/lib/<os>-<arch>/.
#   ./build_cuda.sh            -> libauralis_cuda.so     (GPU, nvcc)
#   ./build_cuda.sh --emulate  -> libauralis_cuda_emu.so (CPU, g++, no GPU needed)
# Architecture: -arch=native, else sm_89, sm_86, sm_75; force with AUR_CUDA_ARCH=sm_86.
# Requires CUDA >= 11.8 for sm_89 (Ada), and a toolkit no newer than the driver (nvidia-smi).
set -euo pipefail
cd "$(dirname "$0")"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
# the same platform tag as auralis/engine/native.py: linux-aarch64, macos-arm64
ARCH="$(uname -m)"; case "$ARCH" in amd64|x86_64) ARCH=x86_64;; arm64|aarch64) ARCH=aarch64;; esac
case "$OS" in darwin) OS=macos; EXT=dylib; [ "$ARCH" = aarch64 ] && ARCH=arm64;; *) OS=linux; EXT=so;; esac
OUT_DIR="../auralis/lib/${OS}-${ARCH}"
mkdir -p "$OUT_DIR"

if [ "${1:-}" = "--emulate" ]; then
  OUT="$OUT_DIR/libauralis_cuda_emu.$EXT"
  # no FMA contraction: the emulation is the bit-exact reference of the split runs, and aarch64
  # contracts a*b+c by default, differently in two loops of the same arithmetic
  g++ -O2 -std=c++17 -shared -fPIC -pthread -ffp-contract=off -DAUR_EMU -x c++ src/aur_cuda.cu -o "$OUT"
  echo "OK (CPU emulation) -> $OUT"
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found: install the CUDA Toolkit (>= 11.8 for Ada" >&2
  echo "cards such as the RTX 2000 Ada), or use --emulate." >&2
  echo "Under conda:  conda install -c nvidia cuda-toolkit=12.4" >&2
  exit 1
fi

VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
MAJ="${VER%% *}"; MIN="${VER##* }"
echo "nvcc found: CUDA ${MAJ}.${MIN}"
if [ "${MAJ:-0}" -lt 11 ] || { [ "${MAJ}" = 11 ] && [ "${MIN:-0}" -lt 8 ]; }; then
  if [ -z "${AUR_CUDA_ARCH:-}" ]; then
    echo "" >&2
    echo "ERROR: CUDA ${MAJ}.${MIN} does not know sm_89 (Ada, e.g." >&2
    echo "RTX 2000 Ada) - CUDA >= 11.8 is required, and old toolkits" >&2
    echo "also reject recent gcc versions." >&2
    echo "Fix (without touching the system, from your conda env):" >&2
    echo "  1) nvidia-smi   # read 'CUDA Version' (driver maximum)" >&2
    echo "  2) conda install -c nvidia cuda-toolkit=12.4   # or <= max" >&2
    echo "  3) rerun this script (which nvcc must point into the env)" >&2
    echo "To target an old card with this toolkit anyway:" >&2
    echo "  AUR_CUDA_ARCH=sm_75 ./build_cuda.sh" >&2
    exit 1
  fi
fi

OUT="$OUT_DIR/libauralis_cuda.$EXT"
# -pthread: the multi-GPU code runs one host thread per device
FLAGS=(-O3 -std=c++17 --shared -Xcompiler -fPIC -Xcompiler -pthread)
ERRLOG="$(mktemp)"
trap 'rm -f "$ERRLOG"' EXIT

try_arch() {
  echo ">> nvcc ${FLAGS[*]} $1 src/aur_cuda.cu -o $OUT"
  nvcc "${FLAGS[@]}" $1 src/aur_cuda.cu -o "$OUT" 2>>"$ERRLOG"
}

show_err() {
  echo "----- nvcc output (tail) -----" >&2
  tail -n 20 "$ERRLOG" >&2
  if grep -qi "unsupported GNU version" "$ERRLOG"; then
    echo "" >&2
    echo "-> Your host gcc is too recent for this CUDA toolkit." >&2
    echo "   The real fix is a recent toolkit (see above);" >&2
    echo "   possible but unsupported workaround:" >&2
    echo "   NVCC_APPEND_FLAGS='-allow-unsupported-compiler'" >&2
  fi
}

if [ -n "${AUR_CUDA_ARCH:-}" ]; then
  try_arch "-arch=${AUR_CUDA_ARCH}" || { show_err; exit 1; }
elif try_arch "-arch=native"; then
  # with no GPU visible, -arch=native silently falls back to an old architecture: warn
  if command -v cuobjdump >/dev/null 2>&1; then
    BUILT="$(cuobjdump --list-elf "$OUT" 2>/dev/null | sed -n 's/.*\.\(sm_[0-9]*\)\.cubin/\1/p' | sort -u | tr '\n' ' ')"
    echo "built for: ${BUILT:-unknown}"
    if ! nvidia-smi >/dev/null 2>&1 || [ "${BUILT}" = "sm_52 " ]; then
      echo "WARNING: no usable GPU seen by nvcc: -arch=native gave '${BUILT}'. For the RTX 2000" >&2
      echo "Ada of this machine: AUR_CUDA_ARCH=sm_89 ./build_cuda.sh" >&2
    fi
  fi
else
  ok=0
  for a in sm_89 sm_86 sm_75; do
    if try_arch "-arch=$a"; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "nvcc failed for sm_89/sm_86/sm_75." >&2; show_err; exit 1; }
fi
echo "OK -> $OUT"
echo "Check: python3 -c \"from auralis.engine import gpu_available; print(gpu_available())\""
