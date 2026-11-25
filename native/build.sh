#!/usr/bin/env bash
# Build libauralis with g++ or clang++ into auralis/lib/<os>-<arch>/, where Python finds it.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-}"
if [ -z "$CXX" ]; then
  if command -v g++ >/dev/null 2>&1; then CXX=g++;
  elif command -v clang++ >/dev/null 2>&1; then CXX=clang++;
  else echo "No C++ compiler found (g++ or clang++)." >&2; exit 1; fi
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
# the same platform tag as auralis/engine/native.py: linux-aarch64, macos-arm64
case "$ARCH" in
  amd64|x86_64) ARCH=x86_64 ;;
  arm64|aarch64) ARCH=aarch64 ;;
esac
case "$OS" in
  darwin) OS=macos; OUT_EXT=dylib; EXTRA="-undefined dynamic_lookup"
          [ "$ARCH" = aarch64 ] && ARCH=arm64 ;;
  *)      OS=linux; OUT_EXT=so; EXTRA="" ;;
esac

OUT_DIR="../auralis/lib/${OS}-${ARCH}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/libauralis.$OUT_EXT"

OUT_TMP="$(mktemp -d)"
trap 'rm -rf "$OUT_TMP"' EXIT
set -x
# the grid builder must reproduce the Python reference bit for bit: no FMA contraction
"$CXX" -O3 -std=c++17 -fPIC -pthread -ffp-contract=off \
  -c src/aur_grid.cpp -o "$OUT_TMP/aur_grid.o"
"$CXX" -O3 -std=c++17 -shared -fPIC -pthread \
  src/aur_native.cpp "$OUT_TMP/aur_grid.o" -o "$OUT" $EXTRA
set +x
echo "OK -> $OUT"
