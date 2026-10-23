#!/usr/bin/env python3
"""Build, validate, and benchmark the project's custom CUDA softmax.

Run from the repository root:
    python benchmarks/bench_softmax.py

Requires CUDA-enabled PyTorch, a compatible CUDA toolkit/nvcc, and an NVIDIA driver.
This script intentionally produces no timings if CUDA is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE = ROOT / "cuda_kernels" / "softmax_extension.cpp"
CUDA_SOURCE = ROOT / "cuda_kernels" / "softmax_kernel.cu"


def build_extension(verbose: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run on an NVIDIA GPU with CUDA-enabled PyTorch, "
            "a compatible driver, and nvcc. No timings were produced."
        )
    return load(
        name="lohit_gpu_softmax_ext",
        sources=[str(CPP_SOURCE), str(CUDA_SOURCE)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=verbose,
    )


def event_latency_ms(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def validate(custom: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = (custom - reference).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    row_sums = custom.sum(dim=-1)
    max_sum_error = float((row_sums - 1.0).abs().max().item()) if row_sums.numel() else 0.0
    torch.testing.assert_close(custom, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=2e-5, rtol=2e-5)
    return max_abs, max_sum_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100,
                        help="timed launches per sample (default: 100)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="warmup launches (default: 20)")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("results/softmax_benchmark.csv"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup non-negative")

    try:
        ext = build_extension(args.verbose_build)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    shapes = [(128, 128), (1024, 1024), (1024, 4096), (4096, 4096)]
    scales = (1.0, 10.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print("Results are measured by this run; no values are hard-coded.")

    for rows, cols in shapes:
        for scale in scales:
            generator = torch.Generator(device="cuda").manual_seed(
                1234 + rows + cols + int(scale * 10)
            )
            x = torch.randn((rows, cols), device="cuda", dtype=torch.float32,
                            generator=generator) * scale
            reference = torch.softmax(x, dim=-1)
            custom = ext.softmax_cuda(x)
            torch.cuda.synchronize()
            max_abs_error, max_row_sum_error = validate(custom, reference)

            custom_ms = event_latency_ms(lambda: ext.softmax_cuda(x),
                                         args.warmup, args.iterations)
            torch_ms = event_latency_ms(lambda: torch.softmax(x, dim=-1),
                                        args.warmup, args.iterations)
            record = {
                "kernel": "custom_row_softmax",
                "reference": "torch.softmax_cuda",
                "gpu": torch.cuda.get_device_name(0),
                "dtype": "float32",
                "rows": rows,
                "cols": cols,
                "scale": scale,
                "custom_median_ms": round(custom_ms, 6),
                "torch_median_ms": round(torch_ms, 6),
                "torch_over_custom_speedup": round(torch_ms / custom_ms, 4) if custom_ms else "",
                "max_abs_error": max_abs_error,
                "max_row_sum_error": max_row_sum_error,
                "warmup_launches": args.warmup,
                "timed_launches_per_sample": args.iterations,
                "timing_samples": 5,
            }
            records.append(record)
            print(
                f"rows={rows:4d} cols={cols:4d} scale={scale:4.1f} "
                f"custom={custom_ms:.6f} ms torch={torch_ms:.6f} ms "
                f"max_abs_err={max_abs_error:.3e} row_sum_err={max_row_sum_error:.3e}"
            )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {len(records)} measured rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
