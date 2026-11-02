"""Strict schema helpers for actual benchmark artifacts; no fallback values."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

START_MARKER = "BENCHMARK_JSON_START"
END_MARKER = "BENCHMARK_JSON_END"


class BenchmarkSchemaError(ValueError):
    """Invalid or incomplete benchmark artifact."""


def validate_benchmark(data: dict[str, Any]) -> None:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise BenchmarkSchemaError("Expected a JSON object with integer schema_version=1.")
    if data["schema_version"] != 1:
        raise BenchmarkSchemaError("Unsupported schema_version; expected 1.")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise BenchmarkSchemaError("Missing metadata object.")
    for key in ("gpu_name", "compute_capability"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkSchemaError(f"metadata.{key} must be a non-empty string.")
    for key in ("matrix_size", "warmup_iterations", "timed_iterations"):
        value = metadata.get(key)
        if type(value) is not int or value <= 0:
            raise BenchmarkSchemaError(f"metadata.{key} must be a positive integer.")
    shape = metadata.get("shape")
    if shape is not None:
        if not isinstance(shape, dict):
            raise BenchmarkSchemaError("metadata.shape must be an object.")
        for key in ("M", "N", "K"):
            value = shape.get(key)
            if type(value) is not int or value <= 0:
                raise BenchmarkSchemaError(f"metadata.shape.{key} must be a positive integer.")

    reference = data.get("reference")
    if not isinstance(reference, dict):
        raise BenchmarkSchemaError("Missing reference object.")
    name = reference.get("name")
    if not isinstance(name, str) or not name.strip():
        raise BenchmarkSchemaError("reference.name must be a non-empty string.")
    _positive(reference.get("latency_ms"), "reference.latency_ms")
    _positive(reference.get("gflops"), "reference.gflops")

    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise BenchmarkSchemaError("results must be a non-empty list.")

    seen: set[str] = set()
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise BenchmarkSchemaError(f"results[{index}] must be an object.")
        for key in ("kernel_id", "name", "input_precision", "accumulation_precision"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkSchemaError(
                    f"results[{index}].{key} must be a non-empty string."
                )
        kernel_id = row["kernel_id"]
        if kernel_id in seen:
            raise BenchmarkSchemaError(f"Duplicate kernel_id: {kernel_id}.")
        seen.add(kernel_id)

        _positive(row.get("latency_ms"), f"results[{index}].latency_ms")
        _positive(row.get("gflops"), f"results[{index}].gflops")

        pct_key = "pct_of_reference" if "pct_of_reference" in row else "pct_of_cublas"
        if pct_key in row:
            _nonnegative(row[pct_key], f"results[{index}].{pct_key}")

        validation = row.get("validation")
        if validation is not None:
            if not isinstance(validation, dict) or type(validation.get("passed")) is not bool:
                raise BenchmarkSchemaError(
                    f"results[{index}].validation must include boolean passed."
                )
            _nonnegative(
                validation.get("frobenius_relative_error"),
                f"results[{index}].validation.frobenius_relative_error",
            )
            _nonnegative(
                validation.get("max_absolute_error"),
                f"results[{index}].validation.max_absolute_error",
            )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkSchemaError(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise BenchmarkSchemaError(f"{field} must be a finite number.")
    return number


def _positive(value: Any, field: str) -> None:
    if _finite_number(value, field) <= 0:
        raise BenchmarkSchemaError(f"{field} must be positive.")


def _nonnegative(value: Any, field: str) -> None:
    if _finite_number(value, field) < 0:
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
