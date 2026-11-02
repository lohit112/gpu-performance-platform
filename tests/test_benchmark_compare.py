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


def _write_comparison_fixture(path, *, gpu="Test GPU", input_precision="FP32"):
    data = {
        "schema_version": 1,
        "metadata": {
            "backend": "test",
            "gpu_name": gpu,
            "compute_capability": "7.5",
            "matrix_size": 64,
            "shape": {"M": 64, "N": 64, "K": 64},
            "layout": "row_major",
            "warmup_iterations": 2,
            "timed_iterations": 10,
            "compiler": "test-compiler",
            "compiler_version": "1.2.3",
            "source_sha256": "abc123",
        },
        "reference": {
            "name": "reference",
            "input_precision": "FP32",
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": 1.0,
            "gflops": 1.0,
        },
        "results": [{
            "kernel_id": "K0",
            "name": "test GEMM",
            "input_precision": input_precision,
            "accumulation_precision": "FP32",
            "output_precision": "FP32",
            "latency_ms": 2.0,
            "gflops": 0.5,
            "validation": {
                "frobenius_relative_error": 0.0,
                "max_absolute_error": 0.0,
                "passed": True,
            },
        }],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_compare_marks_hardware_mismatch(tmp_path):
    first = tmp_path / "t4.json"
    second = tmp_path / "a100.json"
    _write_comparison_fixture(first, gpu="Tesla T4")
    _write_comparison_fixture(second, gpu="A100")

    rows = collect_rows([first, second])

    assert len(rows) == 2
    assert all(row["run_compatibility"].startswith("MISMATCH: GPU differs") for row in rows)


def test_compare_separates_mixed_precision_kernel_groups(tmp_path):
    fp32 = tmp_path / "fp32.json"
    fp16 = tmp_path / "fp16.json"
    _write_comparison_fixture(fp32, input_precision="FP32")
    _write_comparison_fixture(fp16, input_precision="FP16")

    rows = collect_rows([fp32, fp16])

    assert rows[0]["run_compatibility"] == "MATCHED"
    assert rows[0]["comparison_group"] != rows[1]["comparison_group"]


def test_comparison_csv_preserves_toolchain_metadata(tmp_path):
    artifact = tmp_path / "toolchain.json"
    _write_comparison_fixture(artifact)
    rows = collect_rows([artifact])

    assert '"compiler": "test-compiler"' in rows[0]["toolchain"]
    assert '"source_sha256": "abc123"' in rows[0]["toolchain"]
