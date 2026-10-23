# GEMM Benchmark Audit — Required Before Publishing Performance Claims

## What the public repository currently says

The README's previous table reported a T4 cuBLAS FP32 reference of 44,100 GFLOP/s, K0-to-K4 speedup of 508x, and K4 WMMA at 33,000 GFLOP/s. These values should be treated as **historical and unverified until the new validation driver is run**.

NVIDIA's published Tesla T4 specification lists peak single-precision (FP32) performance of 8.1 TFLOP/s and mixed FP16/FP32 Tensor Core performance of 65 TFLOP/s. A 44.1 TFLOP/s `cublasSgemm` result on a T4 is inconsistent with that FP32 specification and strongly indicates a stale result, timing/parser mistake, device mismatch, or incorrect labeling. Official source: https://www.nvidia.com/en-in/data-center/tesla-t4/ .

## Precision-comparison issue

- K0–K3 use FP32 inputs and produce FP32 outputs.
- K4 converts the input matrices to FP16, uses WMMA with FP16 inputs and FP32 accumulation, and stores FP32 results.
- Comparing K4's throughput to an FP32 `cublasSgemm` number is not an apples-to-apples efficiency metric.
- The new validator uses `cublasSgemm` as the K0–K3 reference and `cublasGemmEx` with FP16 inputs / FP32 accumulation for K4. The CSV labels these paths independently.

## Rerun protocol

1. Record `nvidia-smi`, CUDA toolkit, cuBLAS version, GPU model, compute capability, power/clock state when available, and the exact command.
2. Use seeded input generation. Check allocation and API return codes.
3. Warm up each kernel. Use CUDA events and repeat at least 5 timing batches; publish median and range, not a single average.
4. Validate each custom output against the matching precision reference. Record relative Frobenius error and maximum absolute error with documented dtype-appropriate tolerances.
5. Separate conversion time from kernel-only timing, and report end-to-end latency separately when conversions are part of the intended workload.
6. Use the CSV from `matmul_validation` as the source of performance tables. Don't populate machine-readable fields with explanatory placeholder text.
7. The existing kernels are only validated at the fixed `N=4096` here. Do not claim arbitrary dimensions until tail/edge paths are tested.
8. Capture real Nsight Compute data from an environment where `ncu` can attach to the actual GPU. Demo counters must remain labelled as demo data.

## Resume/README policy

Until a clean run exists, remove numerical headline tables or label old data historical/unverified. Do not claim 44,100 GFLOP/s FP32 on a T4. Do not claim same-precision speedup for WMMA vs FP32 cuBLAS.
