# GPU Performance Engineering Platform

CUDA kernel benchmarking, roofline modeling, Nsight Compute profiling, and LLM inference analysis — built on T4/P100 GPU.

![Python](https://img.shields.io/badge/Python-3.10+-3d9abf?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-e8503a?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-f5c842?style=flat-square)

---

## What this is

This project started as a CUDA learning exercise and grew into a small benchmarking and analysis platform. The goal was to understand GPU performance at the level NVIDIA engineers think about it — not just "run a benchmark and report GFLOPS," but trace *why* a kernel performs the way it does and what the hardware is actually doing.

The architecture follows the reasoning loop I found most useful:

```
Hardware Limits → Roofline Model → Kernel Profiling → Optimization → Validation
```

---

## Known Limitations

This project focuses on CUDA optimization techniques rather than reproducing state-of-the-art GEMM implementations.

Current limitations:

- `cp.async` (software pipelining for Ampere+) is not implemented in K3
- K3 vectorizes global loads but computation remains scalar; a full register-tiled + vectorized kernel would score higher
- Benchmarks validated primarily on NVIDIA T4; sm_86/sm_90 behavior may differ
- Nsight Compute profiling (`ncu --csv`) is blocked on Colab T4 due to libcuda stub restrictions in virtualized environments — the nsight module ships with calibrated demo data and a real-CSV path for environments where `ncu` works (Kaggle P100 works; bare metal works)
- Hopper-specific features (TMA, WGMMA) are outside the current scope
- LLM latency estimates assume MFU=0.35 (reasonable for PyTorch + FlashAttention); real numbers vary ±20%

---

## Benchmark Status (Revalidation In Progress)

The previous headline performance tables have been removed while the measurement path is revalidated. The old T4 table reported a 44,100 GFLOP/s cuBLAS FP32 result. NVIDIA's published T4 specification is 8.1 TFLOP/s peak FP32 and 65 TFLOP/s mixed FP16/FP32 Tensor Core performance, so the old FP32 result is not credible as labelled. See [GEMM benchmark audit](docs/GEMM_BENCHMARK_AUDIT.md).

The new validation runner compares K0–K3 against cuBLAS FP32 and K4 against a separate FP16-input / FP32-accumulation cuBLAS reference. It checks numerical error before timing and emits a CSV with GPU identity, dtype, median/min/max timing and correctness status. **No new performance claim is made until this runner compiles and passes on a real NVIDIA GPU.**

The feature branch already includes the small `GPU_PERF_LIBRARY_ONLY` guard needed to reuse the kernel source. Do not apply that patch again when testing this branch.

To run the validation driver, see [GEMM validation instructions](docs/GEMM_VALIDATION_DRIVER.md).

---

## Changelog

### v2

- Replaced static benchmark tables with real benchmark execution via subprocess runner
- Added dynamic parsing of `matmul_bench` output — no hardcoded GFLOPS values
- Implemented true `float4` vectorized loads in K3 (original version described them in comments but didn't use the `float4` type)
- Added FP32→FP16 conversion path for WMMA kernel
- Improved ncu CSV parser to handle metric name variations across ncu versions

---

## Modules

### 1. `cuda_kernels/matmul_suite.cu` — CUDA Kernel Engineering

Five progressively optimized GEMM kernels for N=4096, each targeting a different data-reuse or compute mechanism:

- **K0:** naive output-per-thread baseline
- **K1:** shared-memory tiled GEMM
- **K2:** register-tiled GEMM
- **K3:** vectorized `float4` loads/stores
- **K4:** WMMA Tensor Core path using FP16 inputs and FP32 accumulation

Per-kernel performance values are intentionally withheld until the precision-matched validator in `cuda_kernels/matmul_validation.cu` produces a passing, reproducible CSV. See [the benchmark audit](docs/GEMM_BENCHMARK_AUDIT.md).

```bash
nvcc -O3 -arch=sm_75 -lcublas cuda_kernels/matmul_suite.cu -o matmul_bench
./matmul_bench

# Profile with Nsight Compute:
ncu --set full -o matmul_profile ./matmul_bench
```

---

### 2. `roofline/roofline_model.py` — Roofline Model

The module establishes hardware ceilings from calibration kernels (not spec sheets), then places each kernel at its measured `(FLOPs/Byte, GFLOPS)` operating point.

The key distinction from naive "arithmetic intensity" calculations:

**What most student projects do:**
```python
arithmetic_intensity = fp32_tflops / bandwidth_gbps  # hardware ridge point, not a workload analysis
```

**What this module does:**
1. Establishes hardware ceilings from calibration kernels (not spec sheets)
2. Collects per-kernel FLOPs + DRAM bytes (via `ncu` metrics)
3. Places each kernel at `(FLOPs/Byte, GFLOPS)` on the chart
4. Classifies as compute/memory/mixed bound with ±20% mixed zone around the ridge
5. Generates specific optimization advice based on distance from ceiling

Demo workload values are calibrated to be physically consistent with T4 hardware — every `achieved_gflops` satisfies `achieved ≤ min(OI × bandwidth, peak_compute)`.

```python
from roofline.roofline_model import HardwareRoof, build_demo_workloads, plot_roofline

roofs = [HardwareRoof.t4_measured(), HardwareRoof.a100_spec(), HardwareRoof.h100_spec()]
plot_roofline(roofs, build_demo_workloads(), 'roofline.png')
```

---

### 3. `profiling/nsight_profiler.py` — Nsight Compute Integration

Parses `ncu --csv` output and extracts the 12 metrics that actually matter across 5 bottleneck categories:

| Metric | Category | What it tells you |
|--------|----------|------------------|
| `sm__throughput` | Compute | Are SMs busy? <50% → something else is limiting |
| `stall_long_scoreboard` | Memory | % cycles waiting for L2/HBM — the #1 memory-bound signal |
| `stall_math_throttle` | Compute | % cycles on FP pipeline — the "good" stall |
| `stall_barrier` | Sync | `__syncthreads()` overhead; >20% → try double-buffering |
| `pipe_tensor_cycles_active` | Tensor Cores | TC utilization; 0% on FP16 = missed opportunity |
| `l1_hit_rate / l2_hit_rate` | Cache | Tiling effectiveness |

Rule-based recommendation engine maps each pattern to a specific fix:
- High `stall_long_scoreboard` → add tiling, increase cache reuse
- Low occupancy + high `registers_per_thread` → add `__launch_bounds__`
- High `stall_barrier` → double-buffer with async pipeline (Ampere+)
- Low `tensor_core_pct` on FP16 → switch to WMMA API

```python
# With real ncu output:
report = NsightReport.from_csv('ncu_raw.csv')

# Without GPU (demo mode with calibrated synthetic data):
report = NsightReport._demo_report()
visualize_nsight_report(report, 'nsight_analysis.png')
```

**Note on ncu availability:** Profiling with `ncu` requires a real CUDA driver (not the stub used in Colab). Kaggle's P100 tier and bare-metal instances work fine. The demo mode is explicitly labeled in all outputs.

---

### 4. `llm_profiler/inference_profiler.py` — LLM Inference Analysis

Derives prefill/decode behavior from first principles rather than running inference directly. The core result this module proves from math:

**Model-based inference analysis (not a measured runtime benchmark):** batch-1 decode can become memory-bandwidth-limited depending on model, cache behavior, and serving implementation.

```
Bytes loaded per decode step  = model_weights + KV_cache  (~14 GB for Llama-7B FP16)
FLOPs per decode step         = 2 × num_layers × 4 × hidden_dim²  (~5B FLOPs)
Effective OI                  = 5e9 / 14e9 ≈ 0.36 FLOP/Byte
T4 ridge point (FP16)         = 65 TFLOPS / 280 GB/s ≈ 232 FLOP/Byte

0.36 << 232 → deeply memory-bound, compute is idle
```

The script estimates how batching changes the arithmetic-intensity and memory-traffic picture. Its token-throughput outputs are analytical estimates, not measurements from a running LLM serving stack.

**Supported models:** GPT-2, GPT-2 XL, Llama-2 7B/13B/70B, Llama-3 8B/70B, Mistral 7B, Mixtral 8×7B, GPT-4 (estimate)

**Supported GPUs:** T4, L4, A100-40/80, H100 SXM/PCIe, RTX 4090, RTX 5090, B200

**Outputs:** VRAM breakdown (weights + KV cache + activations), prefill/decode latency, tokens/sec, cost per 1M tokens, bottleneck classification

---

### 5. `report_gen/report_generator.py` — Automated Report Generation

Runs all analyses and generates Markdown + HTML + JSON:

```bash
python main.py --module report
# → report_output/gpu_research_report.md
# → report_output/gpu_research_report.html
# → report_output/benchmark_results.json
```

---

## New Validation Workloads

### GEMM validation

Apply the small source guard before compiling the new driver:

```bash
git apply patches/0001-guard-legacy-main.patch
mkdir -p results/benchmarks
nvcc -O3 -std=c++17 -arch=sm_75 -lcublas \
  cuda_kernels/matmul_validation.cu -o matmul_validation
./matmul_validation results/benchmarks/gemm_validation.csv
```

Change `sm_75` to match the target GPU. This has not yet been compiled or run in this environment; the generated CSV must pass correctness checks before performance figures are published.

### CUDA softmax

The branch adds a row-wise FP32 softmax extension and comparison script:

```bash
python benchmarks/bench_softmax.py --verbose-build
```

The script requires CUDA-enabled PyTorch, a compatible local CUDA toolkit and `nvcc`. It checks results against `torch.softmax` and records timings to `results/softmax_benchmark.csv`. See [softmax instructions](docs/README_SOFTMAX_INSERT.md).

---

## Quick Start

```bash
pip install torch numpy matplotlib

# Run specific module:
python main.py --module roofline    # roofline chart
python main.py --module nsight      # Nsight analysis (demo data)
python main.py --module llm         # LLM profiler
python main.py --module kernels     # kernel benchmark table

# Full report:
python main.py

# Custom LLM profile:
python main.py --module llm --model llama_70b --gpu H100_SXM
```

### With a GPU

```bash
# Compile CUDA kernels (T4 = sm_75, A100 = sm_80, H100 = sm_90)
nvcc -O3 -arch=sm_75 -lcublas cuda_kernels/matmul_suite.cu -o matmul_bench
./matmul_bench

# Profile with Nsight Compute:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
--csv --page raw ./matmul_bench > ncu_raw.csv

# Use real profile data:
python main.py --module nsight --nsight-csv ncu_raw.csv
```

---

## Project Structure

```
gpu_platform_v2/
├── cuda_kernels/
│   └── matmul_suite.cu          ← K0–K4 GEMM kernels + cuBLAS benchmark
├── profiling/
│   └── nsight_profiler.py       ← ncu CSV parser + visualization + recommendations
├── roofline/
│   └── roofline_model.py        ← Roofline model with workload classification
├── llm_profiler/
│   └── inference_profiler.py    ← First-principles VRAM + latency + cost model
├── report_gen/
│   └── report_generator.py      ← Automated MD/HTML/JSON report generation
├── main.py                      ← Unified entry point
└── requirements.txt
```

---

## Development Notes

The initial version used hardcoded benchmark values to develop the visualization pipeline before having real GPU access. That was replaced with a subprocess-based benchmark runner that executes the compiled CUDA binary and parses its output. The benchmark table in the README reflects real T4 measurements from that runner.

Similarly, the original K3 kernel described `float4` loads in comments but used scalar loads in the actual code. The current version uses the `float4` type for global memory accesses.

Nsight profiling on Colab T4 failed because the environment uses a libcuda stub driver — `ncu` exits with an error about driver version mismatches. Those runs were excluded and the nsight module falls back to calibrated demo data in that environment. Kaggle P100 works correctly with `ncu`.

---

## What an NVIDIA engineer will ask

**"How do you establish the naive GEMM bottleneck?"**
> Start from a theoretical arithmetic-intensity model, then check it against a real Nsight Compute capture and a repeatable measurement. The old README's 65 GFLOP/s figure is historical and unverified in this branch, so it is not used as proof until the new CSV and real counter data are available.

**"Why does tiling reduce memory traffic by 32×?"**
> At source level, a 32×32 output tile can reuse a 32×32 input tile of A and B across 32 multiply-accumulate steps. That describes potential reuse, not a measured 32× reduction in DRAM traffic. Use Nsight Compute counters to quantify actual memory traffic and cache behavior.

**"How do you know LLM decode is memory-bound?"**
> At batch=1, each decode step loads ~14 GB of weights to compute ~5B FLOPs → OI ≈ 0.36 FLOP/Byte. Using the stated nominal values, the compute-to-bandwidth ridge point is approximately 65,000 GFLOP/s ÷ 280 GB/s ≈ 232 FLOP/Byte. A modelled operational intensity of 0.36 FLOP/Byte is far below that point, suggesting memory traffic may dominate the idealized scenario; this is an analytical inference, not a measured utilization result.

**"What does `stall_long_scoreboard` mean in Nsight?"**
> It measures the percentage of warp cycles stalled waiting for a long-latency memory operation — specifically an L2 or HBM access after a cache miss. When this metric exceeds 40%, the kernel is memory-latency-bound. The fix is to add tiling (increase cache reuse) or software prefetching to overlap memory latency with computation.

---

## TODOs

- Add `cp.async` path for sm80+ (Ampere async copy pipeline)
- Benchmark larger tile sizes (64×64, 128×128) to find occupancy/tile tradeoff
- Add CUTLASS comparison — understand the gap between K4 and cuBLAS at the instruction level
- Occupancy tuning via `__launch_bounds__` on K2/K3
- Run `ncu --set full` on real hardware and feed actual counter data into the nsight module

---

## License

MIT — use freely, attribution appreciated.
