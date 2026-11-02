"""Compare normalized benchmark JSON artifacts from different backends."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any
from benchmark.schema import load_benchmark


def collect_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = load_benchmark(path)
        meta = data["metadata"]
        for item in data["results"]:
            rows.append({
                "artifact": str(path),
                "backend": meta.get("backend", "unspecified"),
                "gpu": meta.get("gpu_name", "unknown"),
                "compute_capability": meta.get("compute_capability", "unknown"),
                "matrix_size": meta["matrix_size"],
                "kernel_id": item["kernel_id"],
                "kernel": item["name"],
                "input_precision": item["input_precision"],
                "accumulation_precision": item["accumulation_precision"],
                "latency_ms": item["latency_ms"],
                "gflops": item["gflops"],
                "pct_of_reference": item.get("pct_of_cublas", ""),
                "validation_passed": item.get("validation", {}).get("passed", ""),
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
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
