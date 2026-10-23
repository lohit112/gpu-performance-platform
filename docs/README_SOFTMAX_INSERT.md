# Add a Custom CUDA Softmax Workload

The patch adds a row-wise FP32 CUDA softmax extension and a benchmark against `torch.softmax` on the same NVIDIA GPU. This is an additional workload alongside GEMM, not a claim that it beats PyTorch.

## Files

- `cuda_kernels/softmax_extension.cpp`: Python binding.
- `cuda_kernels/softmax_kernel.cu`: stable softmax kernel that subtracts the maximum value in each row before exponentiation.
- `benchmarks/bench_softmax.py`: builds the extension, validates output against PyTorch, times both paths with CUDA events, and writes CSV.

## Run

Use CUDA-enabled PyTorch and a compatible local CUDA toolkit / `nvcc`, then from the repository root:

```bash
python benchmarks/bench_softmax.py --verbose-build
```

Optional:

```bash
python benchmarks/bench_softmax.py --iterations 200 --warmup 30 \
  --output results/softmax_benchmark.csv
```

The script tests four shape pairs with ordinary and high-magnitude finite logits, asserts numerical agreement and row-sum invariants, measures median latency from five samples, and writes only values measured in the current run.

## Current limitation

The development environment used to prepare this patch has CPU-only PyTorch and no `nvcc`, so the CUDA extension has **not** been compiled or run here. It intentionally fails with an explanatory message when CUDA is unavailable. Do not claim performance until a GPU run succeeds.
