#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BINARY="build/matmul_bench"
OUT_DIR="results/profiling"
KERNEL_REGEX="matmul_.*"
NCU_SET="full"
BENCHMARK_JSON="results/benchmarks/latest.json"
if [ "$#" -ge 1 ] && [ -n "$1" ]; then BINARY="$1"; fi
if [ "$#" -ge 2 ] && [ -n "$2" ]; then OUT_DIR="$2"; fi
if [ "$#" -ge 3 ] && [ -n "$3" ]; then KERNEL_REGEX="$3"; fi
if [ "$#" -ge 4 ] && [ -n "$4" ]; then NCU_SET="$4"; fi
if [ "$#" -ge 5 ] && [ -n "$5" ]; then BENCHMARK_JSON="$5"; fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: Nsight Compute CLI (ncu) is not in PATH." >&2
  exit 2
fi
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: benchmark binary is not executable: $BINARY" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
BINARY="$(realpath "$BINARY")"
OUT_DIR="$(realpath "$OUT_DIR")"
BENCHMARK_JSON="$(realpath -m "$BENCHMARK_JSON")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFIX="$OUT_DIR/"$STAMP"_profile"

GPU_NAME="Unknown GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  GPU_NAME="${GPU_NAME:-Unknown GPU}"
fi
NCU_VERSION="$(ncu --version 2>&1 | head -n 1 || true)"
NCU_VERSION="${NCU_VERSION:-unknown}"

# Nsight may replay kernels to collect counters. Its profile duration is not
# the application's CUDA-event benchmark latency.
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:$KERNEL_REGEX" \
  --set "$NCU_SET" \
  --export "$PREFIX" \
  "$BINARY"

# Nsight Compute appends the .ncu-rep extension to the export prefix.
ncu --import "$PREFIX.ncu-rep" --csv --page raw > "$PREFIX.csv"

python3 -m profiling.capture_metadata \
  --binary "$BINARY" \
  --benchmark-json "$BENCHMARK_JSON" \
  --gpu-name "$GPU_NAME" \
  --kernel-regex "$KERNEL_REGEX" \
  --ncu-set "$NCU_SET" \
  --ncu-version "$NCU_VERSION" \
  --report "$PREFIX.ncu-rep" \
  --csv "$PREFIX.csv" \
  --export-prefix "$PREFIX" \
  --output "$PREFIX.meta.json"

echo "Nsight report:    $PREFIX.ncu-rep"
echo "Raw metrics:      $PREFIX.csv"
echo "Provenance:       $PREFIX.meta.json"
