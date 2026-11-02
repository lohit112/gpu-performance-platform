#!/usr/bin/env bash
set -euo pipefail

ARCH=sm_75
SIZE=4096
WARMUP=5
ITERATIONS=20
if [ "$#" -ge 1 ] && [ -n "$1" ]; then ARCH="$1"; fi
if [ "$#" -ge 2 ] && [ -n "$2" ]; then SIZE="$2"; fi
if [ "$#" -ge 3 ] && [ -n "$3" ]; then WARMUP="$3"; fi
if [ "$#" -ge 4 ] && [ -n "$4" ]; then ITERATIONS="$4"; fi

if [[ ! "$ARCH" =~ ^sm_[0-9]+$ ]]; then
  echo "ERROR: expected architecture such as sm_75" >&2
  exit 2
fi
if ! [[ "$SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SIZE must be positive" >&2
  exit 2
fi
if ! [[ "$WARMUP" =~ ^[1-9][0-9]*$ ]] || ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: WARMUP and ITERATIONS must be positive integers" >&2
  exit 2
fi

mkdir -p build results/benchmarks
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACTS=""

CUDA_JSON="results/benchmarks/cuda_"$ARCH"_n"$SIZE"_"$STAMP".json"
python -m benchmark.runner --arch "$ARCH" --size "$SIZE" \
  --warmup "$WARMUP" --iterations "$ITERATIONS" --output "$CUDA_JSON"
ARTIFACTS="$CUDA_JSON"

if [[ -d third_party/cutlass/include ]]; then
  CUTLASS_TAG=""
  if printenv CUTLASS_ARCH_TAG >/dev/null 2>&1; then
    CUTLASS_TAG="$(printenv CUTLASS_ARCH_TAG)"
  fi
  if [ -z "$CUTLASS_TAG" ]; then
    case "$ARCH" in
      sm_75) CUTLASS_TAG=cutlass::arch::Sm75 ;;
      sm_80) CUTLASS_TAG=cutlass::arch::Sm80 ;;
      sm_90) CUTLASS_TAG=cutlass::arch::Sm90 ;;
    esac
  fi
  if [ -n "$CUTLASS_TAG" ]; then
    CUTLASS_REV="$(git -C third_party/cutlass rev-parse --short HEAD)"
    CUTLASS_BIN="build/cutlass_gemm_"$ARCH"_n"$SIZE
    CUTLASS_JSON="results/benchmarks/cutlass_"$ARCH"_n"$SIZE"_"$STAMP".json"
    nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -DGEMM_N="$SIZE" \
      -DWARMUP="$WARMUP" -DITERATIONS="$ITERATIONS" \
      -DCUTLASS_ARCH_TAG="$CUTLASS_TAG" \
      -DCUTLASS_REVISION=\""$CUTLASS_REV"\" \
      -Ithird_party/cutlass/include cutlass/gemm_fp32.cu -lcublas \
      -o "$CUTLASS_BIN"
    "$CUTLASS_BIN" > "$CUTLASS_JSON.stdout.txt"
    python -c 'import json,sys; from pathlib import Path; from benchmark.schema import extract_embedded_json; p=Path(sys.argv[1]); d=extract_embedded_json(p.with_suffix(p.suffix + ".stdout.txt").read_text()); p.write_text(json.dumps(d,indent=2)+"\n")' "$CUTLASS_JSON"
    ARTIFACTS="$ARTIFACTS $CUTLASS_JSON"
  else
    echo "CUTLASS skipped: set CUTLASS_ARCH_TAG for this architecture."
  fi
else
  echo "CUTLASS skipped: clone NVIDIA CUTLASS into third_party/cutlass."
fi

if python -c 'import torch,triton; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  TRITON_JSON="results/benchmarks/triton_"$ARCH"_n"$SIZE"_"$STAMP".json"
  python triton/matmul.py --size "$SIZE" --warmup "$WARMUP" \
    --iterations "$ITERATIONS" --output "$TRITON_JSON"
  ARTIFACTS="$ARTIFACTS $TRITON_JSON"
else
  echo "Triton skipped: CUDA-enabled PyTorch and Triton are not available."
fi

COMPARE_CSV="results/comparison_"$ARCH"_n"$SIZE"_"$STAMP".csv"
# The generated artifact paths do not contain spaces; splitting is intentional.
python -m benchmark.compare $ARTIFACTS --output "$COMPARE_CSV"
echo "Comparison CSV: $COMPARE_CSV"
echo "$ARTIFACTS"
