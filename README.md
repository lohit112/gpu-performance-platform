# GPU Performance Optimization Platform

A reproducible GPU-kernel engineering project for comparing handwritten CUDA,
NVIDIA CUTLASS, and Triton GEMM implementations. The platform combines
numerical validation, CUDA-event timing, machine-readable benchmark artifacts,
Nsight Compute report analysis, and roofline interpretation.

The goal is not to claim that one implementation is always fastest. The goal is
to measure each implementation under controlled conditions, preserve the
precision and hardware context, and use profiler evidence to explain the result.

## Project status

| Component | Current implementation | Evidence status |
|---|---|---|
| Handwritten CUDA GEMM | K0–K4: naive, shared-memory tiled, register tiled, float4, WMMA | Source updated; must be rebuilt and rerun on target GPU |
| Correctness validation | Frobenius-relative error against cuBLAS FP32 reference; precision-specific tolerances | Runs before timing in the CUDA executable; latest source has not yet been hardware-verified |
| Benchmark framework | Compile/run wrapper, strict JSON schema, raw stdout preservation, CSV comparison | CPU-testable Python components; CUDA execution requires NVIDIA GPU and nvcc |
| Nsight Compute | Native report + CSV capture script; parser for detail and raw CSV layouts | Integration implemented; real .ncu-rep captures are not included by default |
| Roofline | Hardware ceiling model and illustrative sample workloads | Demo/analytical only until calibration and DRAM-byte measurements are supplied |
| CUTLASS | Optional FP32 SIMT device::Gemm adapter | Experimental; requires CUTLASS checkout, compilation, and GPU run |
| Triton | Optional tiled FP32 GEMM | Experimental; requires CUDA-enabled PyTorch/Triton and GPU run |
| LLM inference model | First-principles VRAM/latency/throughput estimator | Analytical estimates, not live serving measurements |

**Important:** no benchmark numbers are embedded as fallback values. If CUDA tools or a
GPU are unavailable, the live benchmark command fails with an actionable message
instead of substituting historical estimates.

## Architecture

~~~text
                 GPU PERFORMANCE PLATFORM
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
      CUDA C++             CUTLASS             Triton
       K0–K4             FP32 SIMT           tiled FP32
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                    Benchmark artifact
                 JSON + raw console output
                             |
                +------------+------------+
                |                         |
                v                         v
         Correctness report          Timing results
                |                         |
                +------------+------------+
                             |
                             v
                      CSV comparison
                             |
             +---------------+---------------+
             |                               |
             v                               v
       Nsight Compute                    Roofline
       .ncu-rep + CSV                 measured data only
             |                               |
             +---------------+---------------+
                             |
                             v
                      Report generator
~~~

## CUDA GEMM kernels

The handwritten CUDA suite is in cuda_kernels/matmul_suite.cu.

| Kernel | Technique | Main idea |
|---|---|---|
| K0 — Naive | One thread per output element | Baseline GEMM with no explicit block-level data reuse |
| K1 — Shared-memory tiled | Stage A/B tiles in shared memory | Reuse loaded input tiles across threads in a block |
| K2 — Register tiled | 4×4 output tile per thread | Reuse shared-memory operands across multiple register accumulators |
| K3 — float4 vectorized | 128-bit vector loads/stores | Explore wider global-memory accesses and reduced address/load instruction count |
| K4 — WMMA Tensor Core | FP16 inputs, FP32 accumulation | Use warp-level matrix multiply-accumulate operations |

### Correctness and precision

The executable first computes a cuBLAS FP32 reference, runs each custom kernel,
and calculates:

\[
E_F = \frac{\|C_{\text{kernel}}-C_{\text{reference}}\|_F}
           {\|C_{\text{reference}}\|_F}
\]

K0–K3 use a relative-error tolerance of 1e-4; K4 uses 5e-3 to account for
FP16 input quantization with FP32 accumulation. The benchmark is aborted if any
kernel fails validation.

K4's input precision differs from the FP32 cuBLAS reference. Its reported
GFLOP/s and percentage of FP32 cuBLAS are useful throughput indicators, but
they are **not a precision-matched comparison**. Do not interpret the ratio as
an apples-to-apples numerical efficiency metric. The artifact reports the
FP32-to-FP16 input conversion time separately; K4's kernel latency excludes
that preprocessing pass, so include it when analyzing end-to-end cost.

## Quick start

### Python environment

The general analysis modules require Python, NumPy, and Matplotlib. Install
requirements.txt for the full existing environment, including LLM-model
dependencies. For CPU-only unit tests:

~~~bash
python -m pip install pytest numpy matplotlib
python -m pytest -q
~~~

### Run the live CUDA benchmark

Requirements:

- NVIDIA GPU and compatible driver
- CUDA Toolkit with nvcc
- cuBLAS
- target architecture matching the GPU

For a Tesla T4 (compute capability 7.5):

~~~bash
python -m benchmark.runner --arch sm_75 --size 4096
~~~

Equivalent direct compile command:

~~~bash
mkdir -p build
nvcc -O3 -lineinfo -arch=sm_75 -DN=4096 \
  cuda_kernels/matmul_suite.cu -lcublas -o build/matmul_bench
./build/matmul_bench
~~~

The runner writes:

- results/benchmarks/latest.json — schema-validated measurements and metadata
- results/benchmarks/latest.stdout.txt — complete executable output
- results/benchmarks/latest.compile-stderr.txt — compiler diagnostics

Select another target architecture and matrix size with --arch and --size.
The handwritten CUDA suite requires N to be divisible by 16 because K3 uses
aligned float4 accesses and K4 WMMA operates on complete 16×16 tiles. For
example, an A100 run should use --arch sm_80 on an actual A100 machine.
Only publish a result after the executable has compiled, completed validation,
and run on that hardware.

### Compare CUDA, CUTLASS, and Triton artifacts

Once each backend has produced a validated JSON file:

~~~bash
python -m benchmark.compare \
  results/benchmarks/cuda_t4.json \
  results/benchmarks/cutlass_t4.json \
  results/benchmarks/triton_t4.json \
  --output results/comparison.csv
~~~

Compare only runs with matching GPU, shape, layout, precision, accumulation
mode, warm-up count, and timing protocol. Preserve each raw JSON artifact.
The comparison utility does not manufacture missing backend results.

To orchestrate all backends available in the current GPU environment:

~~~bash
bash scripts/run_full_benchmark.sh sm_75 4096
~~~

The handwritten CUDA suite is required. CUTLASS runs only when its source is
checked out under third_party/cutlass; Triton runs only when CUDA-enabled
PyTorch and Triton are installed. Optional backends are reported as skipped
when unavailable, never replaced with synthetic results.

## CUTLASS backend

The optional adapter is cutlass/gemm_fp32.cu.

~~~bash
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
mkdir -p build
nvcc -O3 -std=c++17 -lineinfo -arch=sm_75 \
  -Ithird_party/cutlass/include \
  cutlass/gemm_fp32.cu -lcublas -o build/cutlass_gemm_fp32
./build/cutlass_gemm_fp32
~~~

Record the CUTLASS Git commit and CUDA/compiler versions with every result.
This adapter is experimental and has not been compiled or benchmarked as part
of the current branch work.

See cutlass/README.md for the comparison contract.

## Triton backend

Install a CUDA-compatible PyTorch build first, then install the optional Triton
dependency:

~~~bash
python -m pip install -r requirements-optional-triton.txt
~~~

Run:

~~~bash
python triton/matmul.py --size 4096 --warmup 5 --iterations 20 \
  --output results/benchmarks/triton_t4.json
~~~

The implementation compares its FP32 output against torch.matmul, records
Frobenius-relative error, and uses CUDA events for timing. It does not create
results without a CUDA-enabled environment.

See triton/README.md for details.

## Nsight Compute profiling

The script scripts/profile_ncu.sh preserves both a native Nsight report and
its raw CSV export. Nsight Compute may replay kernels to collect counters;
profiling duration is not the same as the application's CUDA-event runtime.

Example: profile only the register-tiled kernel on a Tesla T4:

~~~bash
bash scripts/profile_ncu.sh \
  build/matmul_bench \
  results/profiling \
  matmul_register_tiled \
  full
~~~

The last argument selects an Nsight section set (basic, detailed, or
full, depending on the installed Nsight Compute version). The full set can
be substantially slower and collect many more counters.

To analyze the generated CSV:

~~~bash
python main.py --module nsight \
  --nsight-csv results/profiling/<capture-name>.csv
~~~

The parser supports detail-style rows when Metric Name contains raw metric
identifiers, and raw CSV rows where metric names are columns. It tracks which metrics were
actually collected; missing counters are shown as unavailable rather than as
measured zero. Metric availability varies by architecture and Nsight version.

You can open the native .ncu-rep report with the Nsight Compute GUI.

## Roofline analysis

~~~bash
python main.py --module roofline
~~~

Current roofline workloads are illustrative demo points and hardware ceilings
are configured assumptions. An empirical roofline requires measured compute
and memory-bandwidth ceilings plus per-kernel FLOP counts and measured DRAM
bytes. Do not present the demo chart as a measured T4 roofline.

## LLM inference modeling

~~~bash
python main.py --module llm --model llama_7b --gpu T4
~~~

This module estimates model-weight memory, KV-cache size, prefill/decode
behavior, arithmetic intensity, throughput, and cost from explicit assumptions.
It does not run an inference server and its tokens/s values are estimates.

## Report generation

~~~bash
# Live benchmark is attempted; no fallback numbers are substituted.
python main.py --module kernels --arch sm_75 --size 4096

# Generate report with no Nsight section unless a real CSV is provided.
python main.py --module report --arch sm_75 --size 4096

# Explicit synthetic Nsight demo (for layout/testing only).
python main.py --module nsight --demo-data
~~~

The report distinguishes measured benchmark artifacts, synthetic roofline
examples, Nsight demo fixtures, real Nsight CSV data, and analytical LLM
estimates.

## Repository structure

~~~text
gpu-performance-platform/
├── cuda_kernels/
│   └── matmul_suite.cu
├── benchmark/
│   ├── schema.py
│   ├── runner.py
│   └── compare.py
├── cutlass/
│   └── gemm_fp32.cu
├── triton/
│   └── matmul.py
├── profiling/
│   └── nsight_profiler.py
├── roofline/
│   └── roofline_model.py
├── llm_profiler/
│   └── inference_profiler.py
├── report_gen/
│   └── report_generator.py
├── scripts/
│   └── profile_ncu.sh
├── tests/
└── main.py
~~~

## Roadmap

- [x] Keep CUDA kernels and analysis components in one repository.
- [x] Add correctness validation before CUDA timing.
- [x] Emit a versioned machine-readable CUDA benchmark artifact.
- [x] Add a strict benchmark schema and CPU-testable parser/comparison tests.
- [x] Add experimental CUTLASS FP32 and Triton FP32 adapters.
- [x] Add Nsight native-report and CSV capture workflow.
- [ ] Compile and run the updated CUDA suite on a Tesla T4.
- [ ] Run Compute Sanitizer on the updated CUDA binary.
- [ ] Capture real Nsight reports for K0–K4 and analyze observed bottlenecks.
- [ ] Compile/run CUTLASS and Triton on the same T4 workload.
- [ ] Repeat on A100/H100 and compare architecture behavior.
- [ ] Add empirical roofline calibration and measured DRAM traffic.

## License

MIT — see LICENSE.
