# Implementation Status

## Added in this branch

- Stable row-wise FP32 CUDA softmax exposed through a PyTorch extension.
- Softmax benchmark and correctness checks against `torch.softmax` with CUDA-event timing and CSV output.
- GEMM validation driver with separate FP32 and FP16-input/FP32-accumulation cuBLAS references.
- Documentation explaining the mismatch in historical GEMM numbers and a reproducible validation protocol.

## Not yet verified

- The CUDA sources have not been compiled in this environment: `nvcc` is unavailable and PyTorch is CPU-only.
- No GPU timings or Nsight Compute runs were performed here.
- Historical GEMM numbers have not been replaced with invented values. They should remain withheld until the new validation CSV is produced on real NVIDIA hardware.
