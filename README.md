# GPU Performance Engineering Platform

A CUDA-focused engineering project for studying **GPU kernel optimization, memory behavior, roofline analysis, and LLM inference trade-offs**. It combines custom GEMM kernels with Python tools for profiling analysis, model-based inference estimates, and report generation.

[![CUDA](https://img.shields.io/badge/CUDA-kernel%20development-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-analysis%20tools-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-inference%20modeling-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

GPU performance work is more than measuring elapsed time. Results are meaningful only when the workload, precision, hardware, reference implementation, and measurement method are clearly specified.

This repository explores that workflow through five CUDA GEMM implementations and a set of analysis modules. The project is intended as a transparent engineering portfolio: kernel code, timing workflows, profiling parsers, analytical models, and generated reports are kept distinct so readers can tell **device measurements** from **model estimates** and **demo data**.

### Engineering workflow

```text
Workload definition
        |
        v
Hardware / precision assumptions
        |
        v
CUDA kernel implementation -----> cuBLAS comparison
        |                                  |
        v                                  v
CUDA event timing                  Numerical validation
        |                                  |
        +----------------+-----------------+
                         |
                         v
              Nsight Compute analysis
                         |
                         v
                Roofline interpretation
                         |
                         v
             Reproducible report outputs
```

## What is included

| Component | Purpose | Main files |
|---|---|---|
| CUDA GEMM suite | Compare progressively different matrix-multiplication strategies | `cuda_kernels/matmul_suite.cu` |
| Roofline analysis | Model compute and memory ceilings and reason about arithmetic intensity | `roofline/roofline_model.py` |
| Nsight Compute analysis | Parse profiler CSV data, summarize bottlenecks, and generate visual reports | `profiling/nsight_profiler.py` |
| LLM inference profiler | Estimate memory footprint, KV-cache needs, latency, throughput, and cost from model assumptions | `llm_profiler/inference_profiler.py` |
| Report generation | Export analysis outputs as Markdown, HTML, and JSON | `report_gen/report_generator.py` |
| Unified CLI | Run one module or generate a combined report | `main.py` |

## CUDA kernel progression

The GEMM suite contains five implementations. Each illustrates a different optimization strategy; the table describes the implementation, not a claim that every version is faster on every GPU.

| Kernel | Technique | Main concept |
|---|---|---|
| **K0 — Naive** | One thread computes an output element | Baseline with little explicit data reuse |
| **K1 — Shared-memory tiled** | Reuses input tiles staged in shared memory | Reduce repeated global-memory reads |
| **K2 — Register tiled** | Each thread accumulates a small output tile | Reuse values in registers and reduce shared-memory traffic |
| **K3 — Vectorized** | Uses `float4` memory accesses | Explore wider loads/stores and address-instruction overhead |
| **K4 — WMMA Tensor Core** | FP16 inputs with FP32 accumulation | Explore Tensor Core execution and mixed-precision trade-offs |

The current CUDA executable is configured around a fixed `N=4096` workload. Its edge-condition and precision behavior should be validated before treating the suite as a general-purpose GEMM implementation.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/lohit112/gpu-performance-platform.git
cd gpu-performance-platform
```

### 2. Install the Python dependencies

Use a Python environment compatible with the dependencies in `requirements.txt`.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU-backed features, install a CUDA-enabled build of PyTorch compatible with your NVIDIA driver and toolkit. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 3. Run an analysis module

```bash
# Generate the combined report
python main.py

# Individual modules
python main.py --module roofline
python main.py --module nsight
python main.py --module llm
python main.py --module kernels

# Example analytical profile
python main.py --module llm --model llama_70b --gpu H100_SXM
```

The report generator writes outputs under `report_output/` by default. Use the CLI's `--out-dir` option to choose another location.

> **Important:** if the `kernels` module cannot find `nvcc`, the current CLI falls back to a legacy reference table so the rest of the application can run. That fallback is not a live benchmark and must not be cited as a measurement from the current machine.

## Run the CUDA GEMM executable

A supported NVIDIA GPU, NVIDIA driver, CUDA Toolkit, `nvcc`, and cuBLAS are required.

For a Tesla T4 (compute capability 7.5):

```bash
nvcc -O3 -arch=sm_75 -lcublas cuda_kernels/matmul_suite.cu -o matmul_bench
./matmul_bench
```

Change `sm_75` to match the target GPU architecture. The current source's compile-time configuration uses a fixed matrix size; different shapes should not be assumed to be validated automatically.

### Measurement caveats

- Use warm-up iterations and repeated CUDA-event timings. Report the median and spread, along with the exact GPU, driver, CUDA/cuBLAS versions, matrix size, and data type.
- K0–K3 use FP32 data. K4 uses FP16 inputs and accumulates in FP32. Do not present K4's comparison against an FP32 cuBLAS reference as an apples-to-apples efficiency result; compare each kernel with an appropriate reference at matching precision.
- Verify the output against a trusted implementation and report numerical error before publishing performance claims.
- Include data conversion and end-to-end costs separately when they are part of the real workload.

**Benchmark results are intentionally not summarized here until the measurement and precision-comparison path has been reproduced.** Historical values in earlier versions of this README were not sufficiently consistent to publish as validated results. The next published table should link to raw run output and state the configuration and validation status.

## Nsight Compute profiling

The profiling module can parse a CSV export from Nsight Compute and generate an analysis report.

### Capture a profile

Compile the executable first, then run it under `ncu` on a system where Nsight Compute can access the NVIDIA driver and GPU:

```bash
ncu --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
--csv --page raw -o my_profile ./matmul_bench > ncu_raw.csv
```

### Analyze the CSV

```bash
python main.py --module nsight --nsight-csv ncu_raw.csv
```

The parser focuses on signals such as SM throughput, active warps, cache-sector hit rates, memory-dependency stalls, issue throttling, and Tensor Core activity. Interpret metrics in context; no single counter proves a bottleneck by itself.

> **Demo mode:** when no real `--nsight-csv` file is supplied, the `nsight` module uses demo data. Demo-derived charts are illustrative only. Label them as demo output and never describe their values as hardware measurements.

## Roofline analysis

The roofline module combines hardware roof configurations and workload points to reason about whether a workload is likely compute-bound, memory-bound, or mixed. A defensible empirical roofline requires measured hardware ceilings and measured per-kernel FLOPs and memory traffic. If demo or specification-based values are used, they must be labelled accordingly.

```bash
python main.py --module roofline
```

## LLM inference modeling

The inference profiler is a **first-principles analytical model**, not a live inference-serving benchmark. It estimates factors such as:

- Model-weight and KV-cache memory requirements
- Approximate prefill/decode behavior under selected assumptions
- Whether a workload may be constrained by memory bandwidth or compute
- Batch-size and GPU-configuration trade-offs
- Estimated throughput and cost metrics

Run an example:

```bash
python main.py --module llm --model llama_70b --gpu H100_SXM
```

Estimates depend on assumptions such as precision, context length, batch size, and model FLOPs. They should not be presented as observed tokens-per-second measurements from a running model unless separately benchmarked on a real serving stack.

## Report outputs

The report generator exports the analysis into several formats:

```bash
python main.py --module report
```

Expected report artifacts include:

- `report_output/gpu_research_report.md`
- `report_output/gpu_research_report.html`
- `report_output/benchmark_results.json`

Generated files may reflect measured data, analytical estimates, or demo fixtures depending on the selected module and inputs. Preserve that provenance when sharing a report.

## Repository layout

```text
gpu-performance-platform/
├── cuda_kernels/
│   └── matmul_suite.cu
├── llm_profiler/
│   └── inference_profiler.py
├── profiling/
│   └── nsight_profiler.py
├── roofline/
│   └── roofline_model.py
├── report_gen/
│   └── report_generator.py
├── report_output/
├── main.py
├── requirements.txt
└── README.md
```

## Current limitations

- The GEMM implementation is a learning and analysis suite, not a replacement for production libraries such as cuBLAS or CUTLASS.
- The suite currently uses a fixed matrix-size configuration and should not be assumed to support all edge sizes and shapes without additional testing.
- The K4 WMMA path uses mixed precision; results need a precision-matched reference and explicit accuracy reporting.
- Nsight profiling requires a functioning NVIDIA driver and profiler environment. Virtualized environments may restrict hardware-counter access.
- Demo profiler data and analytical LLM estimates are useful for exercising the analysis pipeline, but they are not empirical GPU benchmark results.
- Performance can change substantially with GPU architecture, clocks, memory state, compiler/toolkit, and launch configuration.

## Roadmap

- [ ] Publish reproducible GEMM results with raw timing logs and precision-matched correctness checks.
- [ ] Validate non-multiple dimensions and document kernel boundary behavior.
- [ ] Attach real Nsight Compute captures and tie optimizations to observed counters.
- [ ] Add a second custom CUDA workload and compare correctness and latency against a trusted reference.
- [ ] Separate live-measurement output from demo and analytical output in generated reports.

## License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) if present in the repository.

---

**Author:** [Lohit Reddy](https://github.com/lohit112) · [GitHub profile](https://github.com/lohit112)
