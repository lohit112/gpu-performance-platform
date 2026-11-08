import json

from profiling.capture_metadata import build_capture_metadata
from profiling.nsight_profiler import NsightMetrics, NsightReport
from report_gen.report_generator import ReportGenerator


def _make_report(tmp_path, *, benchmark_gpu="NVIDIA Tesla T4", capture_gpu=None):
    capture_gpu = capture_gpu or benchmark_gpu
    binary = tmp_path / "matmul_bench"
    binary.write_bytes(b"test-binary-for-provenance")
    import hashlib
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()

    artifact = {
        "schema_version": 1,
        "metadata": {
            "gpu_name": benchmark_gpu,
            "compute_capability": "7.5",
            "matrix_size": 2,
            "shape": {"M": 2, "N": 2, "K": 2},
            "warmup_iterations": 1,
            "timed_iterations": 2,
            "binary_path": "build/matmul_bench",
            "binary_sha256": binary_sha,
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
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("synthetic test CSV placeholder\n", encoding="utf-8")
    report_path = tmp_path / "capture.ncu-rep"

    report = ReportGenerator(
        output_dir=str(tmp_path),
        benchmark_json=str(benchmark_json),
        nsight_csv=str(csv_path),
    )
    report.run_kernel_benchmarks()
    report.nsight_report = NsightReport(
        gpu_name=capture_gpu,
        cuda_version="unknown",
        kernels=[
            NsightMetrics(
                kernel_name="matmul_naive",
                dram_bytes_read=32.0,
                dram_bytes_write=16.0,
                available_metrics=["dram_bytes_read", "dram_bytes_write"],
            )
        ],
    )
    report.nsight_metadata = build_capture_metadata(
        binary_path=binary,
        benchmark_json_path=benchmark_json,
        gpu_name=capture_gpu,
        kernel_regex="matmul_.*",
        ncu_set="full",
        ncu_version="test",
        report_path=report_path,
        csv_path=csv_path,
        export_prefix=tmp_path / "capture",
    )
    return report


def test_report_derives_arithmetic_intensity_only_from_verified_capture(tmp_path):
    report = _make_report(tmp_path)

    report.run_empirical_roofline_analysis()

    assert len(report.empirical_roofline_rows) == 1
    row = report.empirical_roofline_rows[0]
    assert row["gemm_flops"] == 16
    assert row["dram_bytes_total"] == 48.0
    assert abs(row["arithmetic_intensity_flop_per_byte"] - (16.0 / 48.0)) < 1e-12
    assert row["benchmark_gflops"] == 0.16
    assert report.empirical_roofline_chart == str(tmp_path / "roofline_analysis_measured.png")
    assert (tmp_path / "roofline_analysis_measured.png").read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))


def test_report_does_not_infer_intensity_without_both_dram_counters(tmp_path):
    report = _make_report(tmp_path)
    report.nsight_report.kernels[0].available_metrics = ["dram_bytes_read"]

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert report.empirical_roofline_chart is None


def test_report_refuses_to_join_different_gpu_profiles(tmp_path):
    report = _make_report(
        tmp_path,
        benchmark_gpu="NVIDIA A100-SXM4-40GB",
        capture_gpu="NVIDIA Tesla T4",
    )

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert report.empirical_roofline_chart is None
    assert "GPU mismatch" in report.empirical_roofline_note


def test_report_refuses_to_join_different_binary_hashes(tmp_path):
    report = _make_report(tmp_path)
    report.nsight_metadata["capture"]["binary_sha256"] = "f" * 64

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert "Binary mismatch" in report.empirical_roofline_note


def test_report_refuses_to_join_different_matrix_shapes(tmp_path):
    report = _make_report(tmp_path)
    report.nsight_metadata["benchmark"]["shape"] = {"M": 4, "N": 4, "K": 4}

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert "Shape mismatch" in report.empirical_roofline_note


def test_report_requires_ncu_sidecar_metadata(tmp_path):
    report = _make_report(tmp_path)
    report.nsight_metadata = None

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert "sidecar is missing" in report.empirical_roofline_note


def test_report_rejects_a_benchmark_json_changed_after_capture(tmp_path):
    report = _make_report(tmp_path)
    with report.benchmark_json.open("a", encoding="utf-8") as artifact:
        artifact.write("\n")

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert "Benchmark artifact mismatch" in report.empirical_roofline_note
