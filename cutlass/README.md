# CUTLASS GEMM adapter

This directory contains an optional FP32 SIMT GEMM comparison adapter using
the CUTLASS 2.x \`cutlass::gemm::device::Gemm\` interface.

## Get CUTLASS

CUTLASS is not vendored into this repository. Obtain NVIDIA's official source:

\`\`\`bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
git -C third_party/cutlass rev-parse HEAD
\`\`\`

Record the exact CUTLASS commit, CUDA Toolkit version, compiler flags, GPU,
matrix dimensions, precision, and benchmark settings with every published run.

## Build on a Tesla T4 (sm_75)

\`\`\`bash
mkdir -p build
CUTLASS_REVISION="$(git -C third_party/cutlass rev-parse --short HEAD)"
nvcc -O3 -std=c++17 -lineinfo -arch=sm_75 \
  -DCUTLASS_ARCH_TAG=cutlass::arch::Sm75 \
  -DCUTLASS_REVISION=\"$CUTLASS_REVISION\" \
  -Ithird_party/cutlass/include \
  cutlass/gemm_fp32.cu -lcublas \
  -o build/cutlass_gemm_fp32
./build/cutlass_gemm_fp32
\`\`\`

The adapter compares CUTLASS FP32 SIMT GEMM with cuBLAS FP32 SGEMM, validates
the output using Frobenius-relative error, and emits the shared benchmark JSON
schema. It is a distinct executable and should be captured as a separate
artifact from the handwritten CUDA suite.

## Comparison rules

- Match M, N, K, layouts, precision, accumulation type, and output type.
- Keep validation tolerance and reference implementation explicit.
- Report compiler/toolkit versions and exact CUTLASS commit.
- Do not compare a Tensor Core mixed-precision kernel against FP32 SIMT and
  describe the throughput numbers as equivalent numerical work.
- Nsight profiling must be run independently on the target GPU.

## Status

**Experimental adapter — not yet compiled or measured in this repository workflow.**
Do not cite CUTLASS performance until the source is built, run, correctness
validated, and captured on a real GPU.


For an A100 build, use a compatible CUTLASS architecture tag and CUDA target,
for example `-DCUTLASS_ARCH_TAG=cutlass::arch::Sm80 -arch=sm_80`. Validate the
H100 configuration against the installed CUTLASS release before using
`Sm90`; do not assume every release has identical SIMT defaults.
