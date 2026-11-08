"""Create a verifiable provenance sidecar for an Nsight Compute capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.schema import load_benchmark


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_key(name: object) -> str:
    normalized = "".join(
        char for char in str(name or "").casefold() if char.isalnum()
    ).removeprefix("nvidia")
    if not normalized or "unknown" in normalized:
        return ""
    return normalized


def _shape(metadata: dict[str, Any]) -> dict[str, int] | None:
    shape = metadata.get("shape")
    if shape is None:
        size = metadata.get("matrix_size")
        shape = {"M": size, "N": size, "K": size}
    if not isinstance(shape, dict):
        return None
    try:
        result = {key: int(shape[key]) for key in ("M", "N", "K")}
    except (KeyError, TypeError, ValueError):
        return None
    return result if all(value > 0 for value in result.values()) else None


def _precision_manifest(results: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for row in results:
        kernel_id = row.get("kernel_id")
        fields = ("input_precision", "accumulation_precision", "output_precision")
        if not isinstance(kernel_id, str) or not all(
            isinstance(row.get(field), str) and row[field].strip() for field in fields
        ):
            return {}
        manifest[kernel_id] = {field: row[field] for field in fields}
    return manifest


def build_capture_metadata(
    *,
    binary_path: str | Path,
    benchmark_json_path: str | Path,
    gpu_name: str,
    kernel_regex: str,
    ncu_set: str,
    ncu_version: str,
    report_path: str | Path,
    csv_path: str | Path,
    export_prefix: str | Path,
) -> dict[str, Any]:
    """Build a sidecar; unverifiable captures are recorded but never marked valid."""
    binary = Path(binary_path).resolve()
    artifact_path = Path(benchmark_json_path).resolve()
    report = Path(report_path).resolve()
    csv_file = Path(csv_path).resolve()
    export = Path(export_prefix).resolve()

    actual_binary_sha256 = sha256_file(binary) if binary.is_file() else None
    artifact_sha256 = sha256_file(artifact_path) if artifact_path.is_file() else None
    checks = {
        "benchmark_artifact_readable": False,
        "benchmark_artifact_hash_present": artifact_sha256 is not None,
        "binary_sha256_matches": False,
        "gpu_identity_matches": False,
        "shape_recorded": False,
        "precision_manifest_recorded": False,
    }
    errors: list[str] = []
    benchmark_info: dict[str, Any] = {}

    if actual_binary_sha256 is None:
        errors.append(f"Benchmark executable not found: {binary}")
    try:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Benchmark JSON not found: {artifact_path}")
        artifact = load_benchmark(artifact_path)
        checks["benchmark_artifact_readable"] = True
        metadata = artifact["metadata"]
        expected_binary_sha256 = metadata.get("binary_sha256")
        checks["binary_sha256_matches"] = bool(
            actual_binary_sha256
            and isinstance(expected_binary_sha256, str)
            and expected_binary_sha256 == actual_binary_sha256
        )
        if not checks["binary_sha256_matches"]:
            errors.append(
                "The executable SHA-256 does not match metadata.binary_sha256 in the benchmark JSON."
            )

        benchmark_gpu = str(metadata.get("gpu_name", "Unknown GPU"))
        checks["gpu_identity_matches"] = bool(
            _gpu_key(gpu_name)
            and _gpu_key(benchmark_gpu)
            and _gpu_key(gpu_name) == _gpu_key(benchmark_gpu)
        )
        if not checks["gpu_identity_matches"]:
            errors.append(
                f"GPU identity mismatch: benchmark={benchmark_gpu!r}, NCU capture={gpu_name!r}."
            )

        benchmark_shape = _shape(metadata)
        checks["shape_recorded"] = benchmark_shape is not None
        if not checks["shape_recorded"]:
            errors.append("Benchmark JSON does not contain a valid M/N/K shape.")

        precision_manifest = _precision_manifest(artifact["results"])
        checks["precision_manifest_recorded"] = bool(precision_manifest)
        if not checks["precision_manifest_recorded"]:
            errors.append("Benchmark JSON does not contain a complete per-kernel precision manifest.")

        benchmark_info = {
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "binary_path": metadata.get("binary_path"),
            "binary_sha256": expected_binary_sha256,
            "gpu_name": benchmark_gpu,
            "compute_capability": metadata.get("compute_capability"),
            "shape": benchmark_shape,
            "layout": metadata.get("layout"),
            "warmup_iterations": metadata.get("warmup_iterations"),
            "timed_iterations": metadata.get("timed_iterations"),
            "precision_by_kernel": precision_manifest,
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"Benchmark artifact could not be validated: {exc}")

    if not checks["benchmark_artifact_hash_present"]:
        errors.append("Benchmark artifact SHA-256 could not be calculated.")

    command = [
        "ncu", "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:{kernel_regex}",
        "--set", ncu_set, "--export", str(export), str(binary),
    ]
    import_command = [
        "ncu", "--import", str(report), "--csv", "--page", "raw",
    ]
    return {
        "schema_version": 1,
        "capture": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": gpu_name,
            "binary_path": str(binary),
            "binary_sha256": actual_binary_sha256,
            "kernel_regex": kernel_regex,
            "ncu_set": ncu_set,
            "ncu_version": ncu_version,
            "report_path": str(report),
            "csv_path": str(csv_file),
            "ncu_command": command,
            "csv_export_command": import_command,
        },
        "benchmark": benchmark_info,
        "validation": {
            "verified": all(checks.values()),
            "checks": checks,
            "errors": errors,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--kernel-regex", required=True)
    parser.add_argument("--ncu-set", required=True)
    parser.add_argument("--ncu-version", default="unknown")
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--export-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sidecar = build_capture_metadata(
        binary_path=args.binary,
        benchmark_json_path=args.benchmark_json,
        gpu_name=args.gpu_name,
        kernel_regex=args.kernel_regex,
        ncu_set=args.ncu_set,
        ncu_version=args.ncu_version,
        report_path=args.report,
        csv_path=args.csv,
        export_prefix=args.export_prefix,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"NCU provenance sidecar: {output}")
    if sidecar["validation"]["verified"]:
        print("NCU capture provenance: VERIFIED")
    else:
        print("WARNING: NCU capture provenance is UNVERIFIED; empirical roofline will be suppressed.")
        for error in sidecar["validation"]["errors"]:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
