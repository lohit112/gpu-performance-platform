# Benchmark data contract

The benchmark package defines one versioned JSON schema shared by CUDA, CUTLASS,
and Triton executables.

## Run the handwritten CUDA suite

\`\`\`bash
python -m benchmark.runner --arch sm_75 --size 4096 --warmup 5 --iterations 20
\`\`\`

This command requires \`nvcc\`, cuBLAS, an NVIDIA driver, and a supported NVIDIA
GPU. It compiles and executes the current CUDA source. It saves a validated
artifact under \`results/benchmarks/latest.json\` and preserves raw stdout beside
it. It does not synthesize a benchmark if compilation or execution fails.

## Compare artifacts

\`\`\`bash
python -m benchmark.compare \\
  results/benchmarks/cuda_t4.json \\
  results/benchmarks/cutlass_t4.json \\
  results/benchmarks/triton_t4.json \\
  --output results/comparison.csv
\`\`\`

Only compare artifacts with matching GPU, matrix shape, precision, accumulation
type, and measurement conditions. The CSV retains each artifact's path and
hardware metadata so incompatible runs can be identified rather than silently
mixed.

For an orchestrated run, use `bash scripts/run_full_benchmark.sh sm_75 4096 5 20`.
The CUDA backend is mandatory; CUTLASS and Triton are run only when their
dependencies are available in the active environment.

## Schema fields

- \`schema_version\`: currently 1
- \`metadata\`: backend, GPU identity, compute capability, dimensions, warmups,
  timed iterations, compiler/framework versions, and capture timestamp
- \`reference\`: named precision-matched reference timing and throughput
- \`results\`: kernel identity, input/accumulation precision, latency, GFLOPS,
  reference percentage, and correctness result

A validation failure is preserved as \`passed: false\`; it is never relabelled
as success. Missing real measurements remain missing.
