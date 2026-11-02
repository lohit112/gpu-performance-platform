"""Compile and execute CUDA GEMM; save only real validated measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from benchmark.schema import extract_embedded_json


def _run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def run_cuda_benchmark(repo_root: str | Path, *, arch: str = "sm_75",
                       size: int = 4096,
                       output_path: str | Path = "results/benchmarks/latest.json",
                       timeout_seconds: int = 900) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source = root / "cuda_kernels" / "matmul_suite.cu"
    if not source.is_file():
        raise RuntimeError(f"CUDA source not found: {source}")
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        raise ValueError("arch must look like sm_75, sm_80, or sm_90.")
    if size <= 0:
        raise ValueError("size must be a positive integer.")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc was not found. Use a CUDA-enabled environment with an NVIDIA GPU, "
            "driver, CUDA Toolkit, and cuBLAS. No synthetic/reference table is substituted."
        )
    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "matmul_bench"
    flags = ["-O3", "-lineinfo", f"-arch={arch}", f"-DN={size}", "-lcublas"]
    compiled = _run([nvcc, *flags[:-1], str(source), flags[-1], "-o", str(binary)],
                    cwd=root, timeout=timeout_seconds)
    executed = _run([str(binary)], cwd=root, timeout=timeout_seconds)
    data = extract_embedded_json(executed.stdout)
    data.setdefault("metadata", {}).update({
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.platform(),
        "compile_arch": arch,
        "compiler": "nvcc",
        "compiler_flags": flags,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (output.parent / "latest.stdout.txt").write_text(executed.stdout, encoding="utf-8")
    (output.parent / "latest.compile-stderr.txt").write_text(compiled.stderr, encoding="utf-8")
    print(f"Validated benchmark artifact: {output}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_75")
    parser.add_argument("--size", type=int, default=4096,
                        help="Square GEMM dimension N (compiles with -DN=N).")
    parser.add_argument("--output", default="results/benchmarks/latest.json")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_cuda_benchmark(root, arch=args.arch, size=args.size,
                       output_path=args.output, timeout_seconds=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
