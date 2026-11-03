"""Compare normalized benchmark JSON artifacts from different backends."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmark.schema import load_benchmark


def _shape(meta: dict[str, Any]) -> dict[str, int]:
    return meta.get("shape") or {
        "M": meta["matrix_size"],
        "N": meta["matrix_size"],
        "K": meta["matrix_size"],
    }


def _run_compatibility_warnings(artifacts: list[dict[str, Any]]) -> list[str]:
    """Report run-level mismatches before interpreting cross-backend timings."""
    if len(artifacts) < 2:
        return []

    warnings: list[str] = []
    fields = (
        ("GPU", lambda m: (m.get("gpu_name"), m.get("compute_capability"))),
        ("matrix shape", lambda m: _shape(m)),
        ("layout", lambda m: m.get("layout", "unknown")),
        ("warm-up count", lambda m: m.get("warmup_iterations")),
        ("timed iteration count", lambda m: m.get("timed_iterations")),
    )
    metas = [item["metadata"] for item in artifacts]
    for label, getter in fields:
        values = [getter(meta) for meta in metas]
        if any(value != values[0] for value in values[1:]):
            warnings.append(f"{label} differs across artifacts: {values}")
    return warnings


def _comparison_group(meta: dict[str, Any], row: dict[str, Any]) -> str:
    """Stable group key; different precision modes are intentionally separated."""
    shape = json.dumps(_shape(meta), sort_keys=True, separators=(",", ":"))
    precision = "/".join((
        row["input_precision"],
        row["accumulation_precision"],
        row["output_precision"],
    ))
    return " | ".join((
        f"{meta.get('gpu_name', 'unknown')} sm_{meta.get('compute_capability', 'unknown')}",
        shape,
        meta.get("layout", "layout unknown"),
        precision,
        f"warmup={meta['warmup_iterations']}",
        f"timed={meta['timed_iterations']}",
    ))


def collect_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    artifacts = [{"path": str(path), "data": load_benchmark(path)} for path in paths]
    if not artifacts:
        return []

    run_warnings = _run_compatibility_warnings([item["data"] for item in artifacts])
    compatibility = "MATCHED" if not run_warnings else "MISMATCH: " + " | ".join(run_warnings)

    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        data = artifact["data"]
        meta = data["metadata"]
        for item in data["results"]:
            shape = _shape(meta)
            group = _comparison_group(meta, item)
            rows.append({
                "artifact": artifact["path"],
                "backend": meta.get("backend", "unspecified"),
                "gpu": meta.get("gpu_name", "unknown"),
                "compute_capability": meta.get("compute_capability", "unknown"),
                "matrix_size": meta["matrix_size"],
                "shape": json.dumps(shape, sort_keys=True),
                "layout": meta.get("layout", "unknown"),
                "toolchain": json.dumps({
                    key: meta[key] for key in (
                        "compiler", "compiler_version", "compiler_flags",
                        "compile_arch", "cuda_runtime_version", "cuda_driver_version",
                        "torch_version", "torch_cuda_version", "tf32_enabled",
                        "triton_version", "cutlass_revision", "cutlass_arch_tag",
                        "source_sha256",
                    ) if key in meta
                }, sort_keys=True),
                "warmup_iterations": meta["warmup_iterations"],
                "timed_iterations": meta["timed_iterations"],
                "fp16_input_conversion_ms": meta.get("fp16_input_conversion_ms", ""),
                "comparison_group": group,
                "run_compatibility": compatibility,
                "reference_name": data["reference"].get("name", "unknown"),
                "reference_input_precision": data["reference"]["input_precision"],
                "reference_accumulation_precision": data["reference"]["accumulation_precision"],
                "reference_output_precision": data["reference"]["output_precision"],
                "kernel_id": item["kernel_id"],
                "kernel": item["name"],
                "input_precision": item["input_precision"],
                "accumulation_precision": item["accumulation_precision"],
                "output_precision": item["output_precision"],
                "latency_ms": item["latency_ms"],
                "gflops": item["gflops"],
                "pct_of_reference": item.get("pct_of_reference", item.get("pct_of_cublas", "")),
                "validation_passed": item["validation"]["passed"],
            })
    return rows


def write_csv(rows: list[dict[str, Any]], destination: str | Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows supplied.")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--output", default="results/comparison.csv")
    args = parser.parse_args()
    rows = collect_rows(args.artifacts)
    write_csv(rows, args.output)

    warnings = sorted({
        row["run_compatibility"] for row in rows
        if row["run_compatibility"] != "MATCHED"
    })
    if warnings:
        print("WARNING: these runs are not directly comparable:")
        for warning in warnings:
            print("  " + warning)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("Compare kernel rows only when their comparison_group values match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
