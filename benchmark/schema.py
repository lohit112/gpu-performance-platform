"""Strict schema helpers for actual benchmark artifacts; no fallback values."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

START_MARKER = "BENCHMARK_JSON_START"
END_MARKER = "BENCHMARK_JSON_END"

class BenchmarkSchemaError(ValueError):
    """Invalid or incomplete benchmark artifact."""


def validate_benchmark(data: dict[str, Any]) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise BenchmarkSchemaError("Expected a schema_version=1 JSON object.")
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        raise BenchmarkSchemaError("Missing metadata object.")
    for key in ("gpu_name", "compute_capability", "matrix_size",
                "warmup_iterations", "timed_iterations"):
        if key not in meta:
            raise BenchmarkSchemaError(f"Missing metadata.{key}.")
    for key in ("matrix_size", "warmup_iterations", "timed_iterations"):
        if not isinstance(meta[key], int) or meta[key] <= 0:
            raise BenchmarkSchemaError(f"metadata.{key} must be a positive integer.")
    ref = data.get("reference")
    if not isinstance(ref, dict):
        raise BenchmarkSchemaError("Missing reference object.")
    _positive(ref.get("latency_ms"), "reference.latency_ms")
    _positive(ref.get("gflops"), "reference.gflops")
    rows = data.get("results")
    if not isinstance(rows, list) or not rows:
        raise BenchmarkSchemaError("results must be a non-empty list.")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkSchemaError(f"results[{i}] must be an object.")
        required = ("kernel_id", "name", "input_precision",
                    "accumulation_precision", "latency_ms", "gflops")
        for key in required:
            if key not in row:
                raise BenchmarkSchemaError(f"results[{i}] is missing {key}.")
        kid = str(row["kernel_id"])
        if kid in seen:
            raise BenchmarkSchemaError(f"Duplicate kernel_id: {kid}.")
        seen.add(kid)
        _positive(row["latency_ms"], f"results[{i}].latency_ms")
        _positive(row["gflops"], f"results[{i}].gflops")
        if "pct_of_cublas" in row and (
            not isinstance(row["pct_of_cublas"], (int, float))
            or row["pct_of_cublas"] < 0
        ):
            raise BenchmarkSchemaError(f"results[{i}].pct_of_cublas must be non-negative.")
        validation = row.get("validation")
        if validation is not None:
            if not isinstance(validation, dict) or not isinstance(validation.get("passed"), bool):
                raise BenchmarkSchemaError(f"results[{i}].validation.passed must be boolean.")
            _nonnegative(validation.get("frobenius_relative_error"),
                         f"results[{i}].validation.frobenius_relative_error")
            _nonnegative(validation.get("max_absolute_error"),
                         f"results[{i}].validation.max_absolute_error")


def _positive(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise BenchmarkSchemaError(f"{field} must be a positive number.")


def _nonnegative(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        raise BenchmarkSchemaError(f"{field} must be non-negative.")


def extract_embedded_json(stdout: str) -> dict[str, Any]:
    if stdout.count(START_MARKER) != 1 or stdout.count(END_MARKER) != 1:
        raise BenchmarkSchemaError("Expected exactly one benchmark JSON marker block.")
    start = stdout.index(START_MARKER) + len(START_MARKER)
    end = stdout.index(END_MARKER)
    if end <= start:
        raise BenchmarkSchemaError("Benchmark JSON markers are out of order.")
    try:
        payload = json.loads(stdout[start:end].strip())
    except json.JSONDecodeError as exc:
        raise BenchmarkSchemaError(f"Invalid JSON in benchmark output: {exc}") from exc
    validate_benchmark(payload)
    return payload


def load_benchmark(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkSchemaError(f"Invalid JSON in {source}: {exc}") from exc
    validate_benchmark(data)
    return data
