from profiling.nsight_profiler import NsightMetrics, NsightReport


def test_detail_csv_layout_parses_named_metric_rows(tmp_path):
    csv_path = tmp_path / "detail.csv"
    csv_path.write_text(
        "Kernel Name,Metric Name,Metric Value\n"
        "matmul_register_tiled,sm__throughput.avg.pct_of_peak_sustained_elapsed,72.5\n"
        "matmul_register_tiled,sm__warps_active.avg.pct_of_peak_sustained_active,48.0\n",
        encoding="utf-8",
    )
    report = NsightReport.from_csv(str(csv_path))
    assert len(report.kernels) == 1
    kernel = report.kernels[0]
    assert kernel.sm_throughput_pct == 72.5
    assert kernel.achieved_occupancy_pct == 48.0
    assert "sm_throughput_pct" in kernel.available_metrics


def test_raw_csv_layout_parses_metric_columns(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "Kernel Name,Device Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed\n"
        "matmul_register_tiled,NVIDIA Tesla T4,72.5,61.2\n"
        "Units,%,%\n",
        encoding="utf-8",
    )
    report = NsightReport.from_csv(str(csv_path))
    kernel = report.kernels[0]
    assert report.gpu_name == "NVIDIA Tesla T4"
    assert kernel.sm_throughput_pct == 72.5
    assert kernel.dram_throughput_pct == 61.2
    assert "dram_throughput_pct" in kernel.available_metrics


def test_missing_csv_raises_instead_of_falling_back_to_demo(tmp_path):
    missing = tmp_path / "missing.csv"
    try:
        NsightReport.from_csv(str(missing))
    except FileNotFoundError:
        return
    raise AssertionError("A missing real profile must not silently use demo data.")


def test_unknown_metric_csv_is_rejected(tmp_path):
    csv_path = tmp_path / "unknown.csv"
    csv_path.write_text(
        "Kernel Name,Some.Unrecognized.Metric\n"
        "matmul_register_tiled,123\n",
        encoding="utf-8",
    )
    try:
        NsightReport.from_csv(str(csv_path))
    except ValueError as exc:
        assert "none of the configured metric columns matched" in str(exc)
        return
    raise AssertionError("CSV with no mapped metrics must be rejected.")


def test_zero_valued_collected_counter_is_still_available(tmp_path):
    csv_path = tmp_path / "zero.csv"
    csv_path.write_text(
        "Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed\n"
        "matmul_wmma,0.0\n",
        encoding="utf-8",
    )
    report = NsightReport.from_csv(str(csv_path))
    metric = report.kernels[0]
    assert metric.sm_throughput_pct == 0.0
    assert "sm_throughput_pct" in metric.available_metrics


def test_registers_per_thread_metric_is_parsed(tmp_path):
    csv_path = tmp_path / "registers.csv"
    csv_path.write_text(
        "Kernel Name,launch__registers_per_thread\n"
        "matmul_register_tiled,64\n",
        encoding="utf-8",
    )
    report = NsightReport.from_csv(str(csv_path))
    kernel = report.kernels[0]
    assert kernel.registers_per_thread == 64
    assert "registers_per_thread" in kernel.available_metrics


def test_high_long_scoreboard_without_high_dram_is_not_called_bandwidth_bound():
    kernel = NsightMetrics(
        kernel_name="matmul_naive",
        stall_long_scoreboard=75.0,
        dram_throughput_pct=25.0,
        available_metrics=["stall_long_scoreboard", "dram_throughput_pct"],
    )
    report = NsightReport(gpu_name="Test GPU", cuda_version="unknown", kernels=[kernel])
    recs = report.generate_recommendations(kernel)
    assert any("Memory-latency/dependency pressure" in item for item in recs)
    assert not any("DRAM-bandwidth constrained" in item for item in recs)


def test_high_long_scoreboard_without_dram_counter_states_uncertainty():
    kernel = NsightMetrics(
        kernel_name="matmul_naive",
        stall_long_scoreboard=75.0,
        available_metrics=["stall_long_scoreboard"],
    )
    report = NsightReport(gpu_name="Test GPU", cuda_version="unknown", kernels=[kernel])
    recs = report.generate_recommendations(kernel)
    assert any("DRAM throughput was not collected" in item for item in recs)
