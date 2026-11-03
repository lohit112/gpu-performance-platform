from profiling.nsight_profiler import NsightMetrics, NsightReport
from report_gen.report_generator import ReportGenerator


def test_report_derives_arithmetic_intensity_from_measured_dram_bytes(tmp_path):
    report = ReportGenerator(
        output_dir=str(tmp_path),
        benchmark_json=None,
        nsight_csv=None,
    )
    report.benchmark_artifact = {
        "metadata": {
            "matrix_size": 2,
            "shape": {"M": 2, "N": 2, "K": 2},
        }
    }
    report.kernel_benchmarks = [{
        "kernel_id": "K0",
        "name": "K0: Naive",
        "input_precision": "FP32",
        "accumulation_precision": "FP32",
        "output_precision": "FP32",
        "latency_ms": 0.1,
        "gflops": 0.16,
    }]
    report.nsight_report = NsightReport(
        gpu_name="NVIDIA Tesla T4",
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

    report.run_empirical_roofline_analysis()

    assert len(report.empirical_roofline_rows) == 1
    row = report.empirical_roofline_rows[0]
    assert row["gemm_flops"] == 16
    assert row["dram_bytes_total"] == 48.0
    assert abs(row["arithmetic_intensity_flop_per_byte"] - (16.0 / 48.0)) < 1e-12
    assert row["benchmark_gflops"] == 0.16
    assert report.empirical_roofline_chart == str(tmp_path / "roofline_analysis_measured.png")
    assert (tmp_path / "roofline_analysis_measured.png").read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))


def test_report_does_not_infer_arithmetic_intensity_without_both_dram_counters(tmp_path):
    report = ReportGenerator(
        output_dir=str(tmp_path),
        benchmark_json=None,
        nsight_csv=None,
    )
    report.benchmark_artifact = {
        "metadata": {
            "matrix_size": 2,
            "shape": {"M": 2, "N": 2, "K": 2},
        }
    }
    report.kernel_benchmarks = [{
        "kernel_id": "K0",
        "name": "K0: Naive",
        "input_precision": "FP32",
        "accumulation_precision": "FP32",
        "output_precision": "FP32",
        "latency_ms": 0.1,
        "gflops": 0.16,
    }]
    report.nsight_report = NsightReport(
        gpu_name="NVIDIA Tesla T4",
        cuda_version="unknown",
        kernels=[
            NsightMetrics(
                kernel_name="matmul_naive",
                dram_bytes_read=32.0,
                available_metrics=["dram_bytes_read"],
            )
        ],
    )

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []



def test_report_refuses_to_join_different_gpu_profiles(tmp_path):
    report = ReportGenerator(
        output_dir=str(tmp_path),
        benchmark_json=None,
        nsight_csv=None,
    )
    report.benchmark_artifact = {
        "metadata": {
            "gpu_name": "NVIDIA A100-SXM4-40GB",
            "matrix_size": 2,
            "shape": {"M": 2, "N": 2, "K": 2},
        }
    }
    report.kernel_benchmarks = [{
        "kernel_id": "K0",
        "name": "K0: Naive",
        "input_precision": "FP32",
        "accumulation_precision": "FP32",
        "output_precision": "FP32",
        "latency_ms": 0.1,
        "gflops": 0.16,
    }]
    report.nsight_report = NsightReport(
        gpu_name="NVIDIA Tesla T4",
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

    report.run_empirical_roofline_analysis()

    assert report.empirical_roofline_rows == []
    assert report.empirical_roofline_chart is None
    assert "GPU mismatch" in report.empirical_roofline_note
