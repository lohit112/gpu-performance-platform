# README edit to make now

The previous README's T4 cuBLAS FP32 number (44,100 GFLOP/s) exceeds the published T4 FP32 peak of 8.1 TFLOP/s. Remove the old measured-result tables until `results/benchmarks/gemm_validation.csv` has been generated and passed correctness checks.

Replace the results block with:

> **Benchmark status:** GEMM kernel results are being revalidated with precision-matched cuBLAS references, numerical error checks, and repeated CUDA-event timing. Previous headline results are withheld until a reproducible run produces raw CSV output. The suite includes FP32 K0–K3 and FP16-input / FP32-accumulation K4, reported separately.

Once the new CSV exists and all rows pass, generate a results table from its rows. Include GPU model, compute capability, dtype, matrix shape, timing spread, reference type, and validation errors.
