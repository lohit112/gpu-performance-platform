#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *T4*) echo "Detected NVIDIA Tesla T4: $GPU_NAME" ;;
  *)
    echo "ERROR: this validation script targets Tesla T4; detected: $GPU_NAME" >&2
    exit 2
    ;;
esac

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is required. Install/enable the CUDA Toolkit." >&2
  exit 2
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "ERROR: compute-sanitizer is required for the memory-safety stage." >&2
  exit 2
fi

mkdir -p build results/profiling results/benchmarks
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Stage 1: run memcheck on a smaller aligned problem. N=256 exercises all
# five kernels and keeps the naive kernel practical under instrumentation.
SANITIZER_BIN="build/matmul_sanitizer_t4"
SANITIZER_LOG="results/profiling/"$STAMP"_compute_sanitizer.txt"
nvcc -O1 -g -lineinfo -arch=sm_75 -DN=256 \
  cuda_kernels/matmul_suite.cu -lcublas -o "$SANITIZER_BIN"
compute-sanitizer --tool memcheck --error-exitcode 1 "$SANITIZER_BIN" \
  2>&1 | tee "$SANITIZER_LOG"
echo "Compute Sanitizer log: $SANITIZER_LOG"

# Stage 2: run all backends installed in this environment on the same 4096 GEMM.
# CUDA is mandatory; CUTLASS and Triton are skipped when their dependencies are absent.
bash scripts/run_full_benchmark.sh sm_75 4096

# Stage 3: collect and parse real Nsight Compute counters when enabled.
RUN_NCU="$(printenv RUN_NCU 2>/dev/null || echo 1)"
if [ "$RUN_NCU" = "1" ]; then
  bash scripts/profile_ncu.sh \
    build/matmul_bench \
    results/profiling \
    'matmul_.*' \
    full
  LATEST_NCU_CSV="$(ls -t results/profiling/*_profile.csv | head -n 1)"
  python main.py --module nsight \
    --nsight-csv "$LATEST_NCU_CSV" \
    --out-dir results/profiling/analysis
  echo "Nsight CSV: $LATEST_NCU_CSV"
else
  echo "Nsight capture skipped because RUN_NCU is not 1."
fi

echo "T4 validation workflow completed."
echo "Benchmark artifacts: results/benchmarks/"
echo "Profiling artifacts: results/profiling/"
