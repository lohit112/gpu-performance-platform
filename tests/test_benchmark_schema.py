import json
import pytest
from benchmark.schema import BenchmarkSchemaError, extract_embedded_json, validate_benchmark


def payload():
    return {
        "schema_version": 1,
        "metadata": {"gpu_name": "Test GPU", "compute_capability": "7.5",
                     "matrix_size": 256, "warmup_iterations": 2, "timed_iterations": 5},
        "reference": {"name": "cuBLAS", "latency_ms": 1.0, "gflops": 10.0},
        "results": [{
            "kernel_id": "K0", "name": "Naive", "input_precision": "FP32",
            "accumulation_precision": "FP32", "latency_ms": 2.0, "gflops": 5.0,
            "pct_of_cublas": 50.0,
            "validation": {"frobenius_relative_error": 1e-7,
                           "max_absolute_error": 1e-5, "passed": True},
        }],
    }


def test_valid_payload():
    validate_benchmark(payload())


def test_extract_embedded_json():
    text = "stdout\nBENCHMARK_JSON_START\n" + json.dumps(payload()) + "\nBENCHMARK_JSON_END\n"
    assert extract_embedded_json(text)["results"][0]["kernel_id"] == "K0"


def test_missing_markers_rejected():
    with pytest.raises(BenchmarkSchemaError):
        extract_embedded_json(json.dumps(payload()))


def test_zero_latency_rejected():
    data = payload()
    data["results"][0]["latency_ms"] = 0
    with pytest.raises(BenchmarkSchemaError):
        validate_benchmark(data)


def test_duplicate_ids_rejected():
    data = payload()
    data["results"].append(dict(data["results"][0]))
    with pytest.raises(BenchmarkSchemaError):
        validate_benchmark(data)


def test_failed_validation_is_preserved():
    data = payload()
    data["results"][0]["validation"]["passed"] = False
    validate_benchmark(data)
    assert data["results"][0]["validation"]["passed"] is False


def test_cuda_runner_refuses_reference_fallback_when_nvcc_missing(tmp_path, monkeypatch):
    from benchmark import runner

    source_dir = tmp_path / "cuda_kernels"
    source_dir.mkdir()
    (source_dir / "matmul_suite.cu").write_text("int main() { return 0; }", encoding="utf-8")
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="No synthetic/reference table"):
        runner.run_cuda_benchmark(tmp_path, arch="sm_75", size=64)


def test_non_finite_benchmark_value_is_rejected():
    data = payload()
    data["results"][0]["gflops"] = float("nan")
    with pytest.raises(BenchmarkSchemaError, match="finite number"):
        validate_benchmark(data)


def test_boolean_latency_is_not_accepted_as_number():
    data = payload()
    data["results"][0]["latency_ms"] = True
    with pytest.raises(BenchmarkSchemaError, match="finite number"):
        validate_benchmark(data)


def test_boolean_schema_version_is_rejected():
    data = payload()
    data["schema_version"] = True
    with pytest.raises(BenchmarkSchemaError):
        validate_benchmark(data)


def test_cuda_runner_rejects_unaligned_matrix_size_before_nvcc(tmp_path, monkeypatch):
    from benchmark import runner

    source_dir = tmp_path / "cuda_kernels"
    source_dir.mkdir()
    (source_dir / "matmul_suite.cu").write_text("int main() { return 0; }", encoding="utf-8")
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)

    with pytest.raises(ValueError, match="divisible by 16"):
        runner.run_cuda_benchmark(tmp_path, arch="sm_75", size=65)
