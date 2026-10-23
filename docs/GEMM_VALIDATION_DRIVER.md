# GEMM Validation Driver

`cuda_kernels/matmul_validation.cu` reuses K0–K4 from `matmul_suite.cu` and provides a separate correctness-first runner. K0–K3 are compared with `cublasSgemm`; K4 is compared with `cublasGemmEx` using FP16 inputs and FP32 accumulation. Each custom output is checked before timing. The CSV includes error metrics, median/min/max timings, throughput and speedup against the matching-precision reference.

## Build/run on a supported NVIDIA GPU

The validator includes `matmul_suite.cu`, so first apply the small guard patch:

```bash
git apply patches/0001-guard-legacy-main.patch
```

Then:

```bash
mkdir -p results/benchmarks
nvcc -O3 -std=c++17 -arch=sm_75 -lcublas \
  cuda_kernels/matmul_validation.cu -o matmul_validation
./matmul_validation results/benchmarks/gemm_validation.csv
```

`sm_75` targets an NVIDIA T4; change the architecture for your actual GPU. The current suite uses fixed N=4096 because the existing WMMA path is not meant to claim arbitrary dimension support. The naive K0 is extremely slow by design and uses one launch per timing sample.

The runner is prepared but not compiled or GPU-validated in this environment. Check build, correctness, and thresholds on real hardware before replacing published results.
