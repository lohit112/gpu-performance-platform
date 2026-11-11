import hashlib
import json

from profiling.capture_metadata import build_capture_metadata


def _write_benchmark(tmp_path, *, gpu_name="NVIDIA Tesla T4"):
    binary = tmp_path / "matmul_bench"
    binary.write_bytes(b"sample-cuda-executable")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1,
        "metadata": {
            "gpu_name": gpu_name,
            "compute_capability": "7.5",
            "matrix_size": 16,
            "shape": {"M": 16, "N": 16, "K": 16},
            "warmup_iterations": 2,
            "timed_iterations": 3,
            "binary_path": "build/matmul_bench",
            "binary_sha256": binary_hash,
        },
        "reference": {
            "name": "cuBLAS FP32",
            "input_precision": "FP32",
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": 0.1,
            "gflops": 0.16,
        },
        "results": [{
            "kernel_id": "K0",
            "name": "K0: Naive",
            "input_precision": "FP32",
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": 0.1,
            "gflops": 0.16,
            "validation": {
                "passed": True,
                "frobenius_relative_error": 0.0,
                "max_absolute_error": 0.0,
            },
        }],
    }
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_json.write_text(json.dumps(artifact), encoding="utf-8")
    return binary, benchmark_json


def _capture(binary, benchmark_json, tmp_path, *, gpu_name="NVIDIA Tesla T4"):
    return build_capture_metadata(
        binary_path=binary,
        benchmark_json_path=benchmark_json,
        gpu_name=gpu_name,
        kernel_regex="matmul_.*",
        ncu_set="full",
        ncu_version="Nsight Compute test version",
        report_path=tmp_path / "profile.ncu-rep",
        csv_path=tmp_path / "profile.csv",
        export_prefix=tmp_path / "profile",
    )


def test_capture_sidecar_verifies_binary_gpu_shape_and_precision(tmp_path):
    binary, benchmark_json = _write_benchmark(tmp_path)

    sidecar = _capture(binary, benchmark_json, tmp_path)

    assert sidecar["schema_version"] == 1
    assert sidecar["validation"]["verified"] is True
    assert all(sidecar["validation"]["checks"].values())
    assert sidecar["benchmark"]["shape"] == {"M": 16, "N": 16, "K": 16}
    assert sidecar["benchmark"]["precision_by_kernel"]["K0"]["input_precision"] == "FP32"
    assert len(sidecar["benchmark"]["artifact_sha256"]) == 64


def test_capture_sidecar_rejects_binary_changed_after_benchmark(tmp_path):
    binary, benchmark_json = _write_benchmark(tmp_path)
    binary.write_bytes(b"different-executable")

    sidecar = _capture(binary, benchmark_json, tmp_path)

    assert sidecar["validation"]["verified"] is False
    assert sidecar["validation"]["checks"]["binary_sha256_matches"] is False
    assert any("executable SHA-256" in item for item in sidecar["validation"]["errors"])


def test_capture_sidecar_rejects_gpu_identity_mismatch(tmp_path):
    binary, benchmark_json = _write_benchmark(tmp_path)

    sidecar = _capture(binary, benchmark_json, tmp_path, gpu_name="NVIDIA A100-SXM4-40GB")

    assert sidecar["validation"]["verified"] is False
    assert sidecar["validation"]["checks"]["gpu_identity_matches"] is False
    assert any("GPU identity mismatch" in item for item in sidecar["validation"]["errors"])
