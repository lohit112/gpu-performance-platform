import sys
import types
from types import SimpleNamespace

import main


def test_failed_live_run_does_not_load_stale_benchmark_json(monkeypatch, tmp_path):
    captured = {}

    class FakeReportGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def export_all(self, primary_gpu):
            return {"markdown": str(tmp_path / "report.md")}

    fake_module = types.ModuleType("report_gen.report_generator")
    fake_module.ReportGenerator = FakeReportGenerator
    monkeypatch.setitem(sys.modules, "report_gen.report_generator", fake_module)

    def fail_live_run(_args):
        raise RuntimeError("nvcc unavailable")

    monkeypatch.setattr(main, "run_kernels", fail_live_run)
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        gpu="T4",
        benchmark_json="results/benchmarks/latest.json",
        nsight_csv=None,
        demo_data=False,
    )

    result = main.run_full_report(args)

    assert captured["benchmark_json"] is None
    assert result["markdown"] == str(tmp_path / "report.md")


def test_successful_live_run_passes_current_artifact_to_report(monkeypatch, tmp_path):
    captured = {}
    artifact = tmp_path / "this_run.json"
    artifact.write_text("{}", encoding="utf-8")

    class FakeReportGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def export_all(self, primary_gpu):
            return {"json": str(tmp_path / "report.json")}

    fake_module = types.ModuleType("report_gen.report_generator")
    fake_module.ReportGenerator = FakeReportGenerator
    monkeypatch.setitem(sys.modules, "report_gen.report_generator", fake_module)

    monkeypatch.setattr(main, "run_kernels", lambda _args: artifact)
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        gpu="T4",
        benchmark_json="old_stale.json",
        nsight_csv=None,
        demo_data=False,
    )

    main.run_full_report(args)

    assert captured["benchmark_json"] == str(artifact)
