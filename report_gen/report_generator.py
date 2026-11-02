"""Generate reports from explicit inputs and preserve result provenance.

CUDA timings are loaded only from validated benchmark JSON. Roofline examples
are analytical illustrations. Nsight metrics are shown only for a real CSV, or
for synthetic demo data when --demo-data was explicitly selected.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from benchmark.schema import load_benchmark
from llm_profiler.inference_profiler import LLMInferenceProfiler, visualize_inference_profile
from roofline.roofline_model import HardwareRoof, build_demo_workloads, plot_roofline, RooflineModel
from profiling.nsight_profiler import NsightReport, visualize_nsight_report


class ReportGenerator:
    def __init__(
        self,
        output_dir: str = "report_output",
        author: str = "GPU Research",
        benchmark_json: str | None = "results/benchmarks/latest.json",
        nsight_csv: str | None = None,
        include_demo_data: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.author = author
        self.benchmark_json = Path(benchmark_json) if benchmark_json else None
        self.nsight_csv = Path(nsight_csv) if nsight_csv else None
        self.include_demo_data = include_demo_data
        self.profiler = LLMInferenceProfiler(mfu_estimate=0.35)
        self.roofline_results = []
        self.roofline_roofs = []
        self.nsight_report = None
        self.llm_profiles = []
        self.kernel_benchmarks = []
        self.benchmark_artifact: dict[str, Any] | None = None

    def run_roofline_analysis(self) -> None:
        print("[1/4] Generating illustrative roofline analysis...")
        roofs = [HardwareRoof.t4_spec(), HardwareRoof.a100_spec(), HardwareRoof.h100_spec()]
        workloads = build_demo_workloads()
        self.roofline_roofs = roofs
        self.roofline_results = RooflineModel(roofs[0]).classify_all(workloads)
        plot_roofline(roofs, workloads, str(self.output_dir / "roofline_analysis_demo.png"))

    def run_nsight_analysis(self) -> None:
        print("[2/4] Loading Nsight Compute data...")
        if self.nsight_csv and self.nsight_csv.is_file():
            self.nsight_report = NsightReport.from_csv(str(self.nsight_csv))
        elif self.include_demo_data:
            self.nsight_report = NsightReport._demo_report()
        else:
            self.nsight_report = None
            print("  No real Nsight CSV supplied; profiler section marked unavailable.")
            return
        visualize_nsight_report(self.nsight_report, str(self.output_dir / "nsight_analysis.png"))
        self.nsight_report.to_json(str(self.output_dir / "nsight_report.json"))

    def run_llm_profiler(self, model: str = "llama_7b", gpu: str = "T4") -> None:
        print("[3/4] Running analytical LLM inference model...")
        self.llm_profiles = self.profiler.batch_profile(
            model, gpu, context_len=4096, precision="fp16"
        )
        visualize_inference_profile(
            self.llm_profiles, str(self.output_dir / "llm_inference_profile.png")
        )

    def run_kernel_benchmarks(self) -> None:
        print("[4/4] Loading validated CUDA benchmark artifact...")
        if self.benchmark_json and self.benchmark_json.is_file():
            self.benchmark_artifact = load_benchmark(self.benchmark_json)
            self.kernel_benchmarks = self.benchmark_artifact["results"]
        else:
            self.benchmark_artifact = None
            self.kernel_benchmarks = []
            print("  No benchmark JSON found; no synthetic GFLOPS values substituted.")

    def generate_markdown(self, primary_gpu: str = "T4") -> str:
        lines = [
            "# GPU Performance Engineering Platform",
            "",
            "## Research Report",
            f"- Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            f"- Requested GPU target: {primary_gpu}",
            f"- Author: {self.author}",
            "",
            "> Provenance: benchmark figures are loaded from a validated executable artifact. "
            "Roofline examples and LLM profiles are analytical; Nsight demo data is synthetic.",
            "",
            "## 1. CUDA GEMM benchmark",
            "",
        ]
        if self.benchmark_artifact:
            meta = self.benchmark_artifact["metadata"]
            ref = self.benchmark_artifact["reference"]
            lines.extend([
                f"- GPU: {meta.get('gpu_name', 'unknown')} "
                f"(compute capability {meta.get('compute_capability', 'unknown')})",
                f"- Matrix size: {meta.get('matrix_size', 'not recorded')}",
                f"- Warm-up iterations: {meta.get('warmup_iterations', 'not recorded')}",
                f"- Timed iterations: {meta.get('timed_iterations', 'not recorded')}",
                f"- Reference: {ref.get('name', 'unknown')} — "
                f"{ref['gflops']:.2f} GFLOPS at {ref['latency_ms']:.4f} ms",
                "",
                "| Kernel | Input | Accumulator | Latency (ms) | GFLOPS | % reference | Correctness |",
                "|---|---|---|---:|---:|---:|---|",
            ])
            for row in self.kernel_benchmarks:
                validation = row.get("validation") or {}
                state = ("PASS" if validation.get("passed") is True else
                         "FAIL" if validation.get("passed") is False else "not reported")
                pct = row.get("pct_of_cublas", row.get("pct_of_reference", ""))
                pct_text = f"{pct:.2f}%" if isinstance(pct, (int, float)) else "n/a"
                lines.append(
                    f"| {row['name']} | {row['input_precision']} | "
                    f"{row['accumulation_precision']} | {row['latency_ms']:.4f} | "
                    f"{row['gflops']:.2f} | {pct_text} | {state} |"
                )
            lines.extend([
                "",
                "Precision note: compare throughput with input and accumulation precision visible. "
                "FP16-input WMMA is not an apples-to-apples numerical comparison with FP32 cuBLAS.",
            ])
        else:
            lines.append(
                "No live CUDA benchmark artifact was supplied. Run "
                "'python -m benchmark.runner --arch sm_75' on a CUDA-enabled GPU."
            )

        lines.extend(["", "## 2. Roofline analysis (illustrative)", ""])
        lines.append(
            "The chart uses configured hardware ceilings and synthetic example workloads. "
            "It is an educational illustration, not an empirical roofline for this run."
        )
        if self.roofline_roofs:
            lines.extend([
                "",
                "| GPU configuration | FP32 ceiling (GFLOPS) | Bandwidth assumption (GB/s) |",
                "|---|---:|---:|",
            ])
            for roof in self.roofline_roofs:
                lines.append(
                    f"| {roof.gpu_name} | {roof.peak_fp32_gflops:.0f} | "
                    f"{roof.peak_bandwidth_gbps:.0f} |"
                )

        lines.extend(["", "## 3. Nsight Compute", ""])
        if self.nsight_report:
            if "demo" in self.nsight_report.gpu_name.lower():
                lines.append("**Synthetic demo data — not a hardware profile.**")
            else:
                lines.append(f"Source CSV: {self.nsight_report.raw_csv_path or 'not recorded'}")
            lines.extend([
                "",
                "| Kernel | SM throughput (%) | Occupancy (%) | L1 hit (%) | L2 hit (%) | Tensor Core (%) |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for metric in self.nsight_report.kernels:
                lines.append(
                    f"| {metric.kernel_name} | {metric.sm_throughput_pct:.2f} | "
                    f"{metric.achieved_occupancy_pct:.2f} | {metric.l1_hit_rate_pct:.2f} | "
                    f"{metric.l2_hit_rate_pct:.2f} | {metric.tensor_core_pct:.2f} |"
                )
        else:
            lines.append(
                "No real Nsight Compute CSV was supplied. Capture one with "
                "scripts/profile_ncu.sh; no profiler metrics are inferred."
            )

        lines.extend(["", "## 4. LLM inference modeling", ""])
        lines.append(
            "The following values are first-principles analytical estimates, not measurements "
            "from a live inference server."
        )
        if self.llm_profiles:
            lines.extend([
                "",
                "| Batch | Model | GPU | Estimated VRAM (GB) | Estimated decode (tokens/s) | Bound |",
                "|---:|---|---|---:|---:|---|",
            ])
            for profile in self.llm_profiles:
                lines.append(
                    f"| {profile.batch_size} | {profile.model.name} | {profile.gpu.name} | "
                    f"{profile.total_vram_gb:.2f} | "
                    f"{profile.decode_tokens_per_sec_batch:.2f} | {profile.decode_bound} |"
                )

        lines.extend([
            "",
            "## 5. Reproducibility and limitations",
            "",
            "- Raw CUDA stdout is stored beside the benchmark JSON artifact.",
            "- Benchmark JSON records GPU identity, compute capability, CUDA versions, matrix size, "
            "warm-ups, timed iterations, precision, latency, throughput, and correctness.",
            "- Nsight output is included only for a real CSV or explicitly selected demo data.",
            "- Roofline examples remain synthetic until hardware ceilings and DRAM bytes are measured.",
            "- LLM values remain estimates until checked against a real inference workload.",
            "",
        ])
        return "\n".join(lines)

    def export_all(self, primary_gpu: str = "T4") -> dict[str, str]:
        self.run_roofline_analysis()
        self.run_nsight_analysis()
        self.run_llm_profiler(gpu=primary_gpu)
        self.run_kernel_benchmarks()

        markdown = self.generate_markdown(primary_gpu)
        md_path = self.output_dir / "gpu_research_report.md"
        md_path.write_text(markdown, encoding="utf-8")
        try:
            import markdown as markdown_lib
            html_body = markdown_lib.markdown(markdown, extensions=["tables", "fenced_code", "toc"])
        except ImportError:
            html_body = "<pre>" + markdown.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>GPU Performance Engineering Report</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;"
            "line-height:1.6;padding:0 20px}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:7px;text-align:left}th{background:#eee}"
            "code{background:#f2f2f2;padding:2px 4px}</style></head><body>"
            + html_body + "</body></html>"
        )
        html_path = self.output_dir / "gpu_research_report.html"
        html_path.write_text(html, encoding="utf-8")

        summary = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "benchmark_source": str(self.benchmark_json) if self.benchmark_artifact else None,
            "kernel_benchmarks": self.kernel_benchmarks,
            "roofline_provenance": "synthetic_demo",
            "nsight_provenance": (
                "synthetic_demo" if self.nsight_report and "demo" in self.nsight_report.gpu_name.lower()
                else "real_csv" if self.nsight_report else "unavailable"
            ),
            "llm_provenance": "analytical_estimate",
            "llm_profiles": [
                {
                    "batch": p.batch_size,
                    "model": p.model.name,
                    "gpu": p.gpu.name,
                    "vram_gb_estimate": p.total_vram_gb,
                    "decode_tokens_per_sec_estimate": p.decode_tokens_per_sec_batch,
                    "decode_bound_estimate": p.decode_bound,
                }
                for p in self.llm_profiles
            ],
        }
        json_path = self.output_dir / "gpu_research_report.json"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return {"markdown": str(md_path), "html": str(html_path), "json": str(json_path)}


if __name__ == "__main__":
    for kind, output in ReportGenerator().export_all().items():
        print(f"{kind}: {output}")
