from profiling.nsight_profiler import NsightReport


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
    assert len(report.kernels) == 1  # the raw CSV units row is not a kernel launch
    kernel = report.kernels[0]
    assert kernel.sm_throughput_pct == 72.5
    assert kernel.achieved_occupancy_pct == 48.0
    assert "sm_throughput_pct" in kernel.available_metrics


def test_raw_csv_layout_parses_metric_columns(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed\n"
        "matmul_register_tiled,72.5,61.2\n"
        "Units,%,%\n",
        encoding="utf-8",
    )
    report = NsightReport.from_csv(str(csv_path))
    kernel = report.kernels[0]
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
