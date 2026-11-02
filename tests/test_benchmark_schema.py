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
