import json
from benchmark.compare import collect_rows, write_csv


def test_compare_artifacts(tmp_path):
    data = {
        "schema_version": 1,
        "metadata": {"gpu_name": "Test GPU", "compute_capability": "7.5",
                     "matrix_size": 64, "warmup_iterations": 1, "timed_iterations": 2},
        "reference": {"name": "ref", "input_precision": "FP32", "accumulation_precision": "FP32", "output_precision": "FP32", "latency_ms": 1.0, "gflops": 1.0},
        "results": [{"kernel_id": "K0", "name": "Naive", "input_precision": "FP32",
                     "accumulation_precision": "FP32", "output_precision": "FP32", "latency_ms": 2.0, "gflops": 0.5,
                     "validation": {"frobenius_relative_error": 0.0,
                                    "max_absolute_error": 0.0, "passed": True}}],
    }
    source = tmp_path / "run.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    rows = collect_rows([source])
    assert len(rows) == 1 and rows[0]["gflops"] == 0.5
    output = tmp_path / "comparison.csv"
    write_csv(rows, output)
    assert "kernel_id" in output.read_text(encoding="utf-8")
