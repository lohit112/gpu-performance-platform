# Triton GEMM backend

The experimental Triton backend implements a tiled FP32 matrix multiplication
and emits the benchmark artifact schema used by the comparison tools.

## Install

Install a CUDA-compatible PyTorch build first, then:

\`\`\`bash
python -m pip install -r requirements-optional-triton.txt
\`\`\`

## Run

\`\`\`bash
python triton/matmul.py --size 4096 --warmup 5 --iterations 20 \\
  --output results/benchmarks/triton_latest.json
\`\`\`

The program compares its result against \`torch.matmul\`, records Frobenius
relative error, times the Triton kernel with CUDA events, and stores GPU and
software metadata. It exits rather than creating results when CUDA is
unavailable or correctness fails.

## Status

**Experimental implementation — not yet run on a target GPU.** Record the
PyTorch/Triton versions, GPU, compute capability, tile configuration, precision,
and results before making performance claims.
