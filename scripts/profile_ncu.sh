#!/usr/bin/env bash
set -euo pipefail

BINARY="build/matmul_bench"
OUT_DIR="results/profiling"
KERNEL_REGEX="matmul_.*"
NCU_SET="full"
if [ "$#" -ge 1 ] && [ -n "$1" ]; then BINARY="$1"; fi
if [ "$#" -ge 2 ] && [ -n "$2" ]; then OUT_DIR="$2"; fi
if [ "$#" -ge 3 ] && [ -n "$3" ]; then KERNEL_REGEX="$3"; fi
if [ "$#" -ge 4 ] && [ -n "$4" ]; then NCU_SET="$4"; fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: Nsight Compute CLI (ncu) is not in PATH." >&2
  exit 2
fi
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: benchmark binary is not executable: $BINARY" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFIX="$OUT_DIR/"$STAMP"_profile"

# Nsight may replay kernels to collect counters. Its profile duration is not
# the application's CUDA-event benchmark latency.
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:$KERNEL_REGEX" \
  --set "$NCU_SET" \
  --export "$PREFIX.ncu-rep" \
  "$BINARY"

# Export the saved native report as raw CSV separately.
ncu --import "$PREFIX.ncu-rep" --csv --page raw > "$PREFIX.csv"

echo "Nsight report: $PREFIX.ncu-rep"
echo "Raw metrics:   $PREFIX.csv"
