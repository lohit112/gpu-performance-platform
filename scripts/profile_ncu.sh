#!/usr/bin/env bash
set -euo pipefail

BINARY="$1"
OUT_DIR="$2"
KERNEL_REGEX="$3"
if [ -z "$BINARY" ]; then BINARY="build/matmul_bench"; fi
if [ -z "$OUT_DIR" ]; then OUT_DIR="results/profiling"; fi
if [ -z "$KERNEL_REGEX" ]; then KERNEL_REGEX="matmul_.*"; fi

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
  --set basic \
  --export "$PREFIX.ncu-rep" \
  "$BINARY"

# Export the saved native report as raw CSV separately.
ncu --import "$PREFIX.ncu-rep" --csv --page raw > "$PREFIX.csv"

echo "Nsight report: $PREFIX.ncu-rep"
echo "Raw metrics:   $PREFIX.csv"
