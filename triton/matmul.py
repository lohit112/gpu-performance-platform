"""Triton FP32 GEMM baseline with correctness and benchmark JSON output.

Requires a CUDA-enabled PyTorch/Triton installation. This module never returns
synthetic results when CUDA is unavailable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BLOCK_K)):
        current_k = k_block * BLOCK_K + offs_k
        a_ptrs = A + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak
        b_ptrs = B + current_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(current_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def run(M: int, N: int, K: int, warmup: int, iterations: int,
        output: str, block_m: int = 32, block_n: int = 64,
        block_k: int = 32) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton benchmark requires a CUDA-enabled PyTorch environment.")
    if min(M, N, K, warmup, iterations) <= 0:
        raise ValueError("Matrix dimensions, warmups, and iterations must be positive.")
    for label, block in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if block <= 0 or block & (block - 1):
            raise ValueError(f"{label} must be a positive power of two.")

    torch.manual_seed(42)
    # Keep the PyTorch reference in strict FP32 mode, matching Triton's ieee dot.
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    reference = torch.matmul(a, b)

    def launch():
        _gemm_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4,
        )

    launch()
    torch.cuda.synchronize()
    diff = (c - reference).to(torch.float64)
    ref64 = reference.to(torch.float64)
    rel_error = (torch.linalg.vector_norm(diff) /
                 torch.clamp(torch.linalg.vector_norm(ref64), min=1e-30)).item()
    max_abs = diff.abs().max().item()
    passed = rel_error <= 1e-4
    if not passed:
        raise RuntimeError(
            f"Triton correctness failed: relative_error={rel_error:.6e}, "
            f"max_abs_error={max_abs:.6e}"
        )

    triton_ms = _time_cuda(launch, warmup, iterations)
    reference_ms = _time_cuda(lambda: torch.matmul(a, b), warmup, iterations)
    flops = 2.0 * M * N * K
    triton_gflops = flops / (triton_ms * 1e6)
    reference_gflops = flops / (reference_ms * 1e6)

    props = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "metadata": {
            "backend": "triton",
            "reference_kind": "torch.matmul",
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "matrix_size": max(M, N, K),
            "shape": {"M": M, "N": N, "K": K},
            "layout": "row_major",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "tf32_enabled": False,
            "triton_version": triton.__version__,
            "block_shape": [block_m, block_n, block_k],
        },
        "reference": {
            "name": "torch.matmul FP32",
            "input_precision": "FP32",
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": reference_ms,
            "gflops": reference_gflops,
        },
        "results": [{
            "kernel_id": "TRITON_FP32",
            "name": "Triton tiled GEMM FP32",
            "input_precision": "FP32",
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": triton_ms,
            "gflops": triton_gflops,
            "pct_of_reference": 100.0 * triton_gflops / reference_gflops,
            "validation": {
                "frobenius_relative_error": rel_error,
                "max_absolute_error": max_abs,
                "passed": passed,
            },
        }],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Triton benchmark saved: {destination}")
    print(f"Latency: {triton_ms:.4f} ms | Throughput: {triton_gflops:.2f} GFLOPS")
    print(f"Correctness: PASS (relative error {rel_error:.3e})")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--output", default="results/benchmarks/triton_latest.json")
    args = parser.parse_args()
    run(args.m or args.size, args.n or args.size, args.k or args.size,
        args.warmup, args.iterations, args.output,
        args.block_m, args.block_n, args.block_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
