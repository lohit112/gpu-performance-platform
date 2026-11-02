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
        self.empirical_roofline_rows: list[dict[str, Any]] = []

    def run_roofline_analysis(self) -> None:
        print("[1/4] Generating illustrative roofline analysis...")
        roofs = [HardwareRoof.t4_spec(), HardwareRoof.a100_spec(), HardwareRoof.h100_spec()]
        workloads = build_demo_workloads()
        self.roofline_roofs = roofs
        self.roofline_results = RooflineModel(roofs[0]).classify_all(workloads)
        plot_roofline(roofs, workloads, str(self.output_dir / "roofline_analysis_demo.png"))

    def run_nsight_analysis(self) -> None:
        print("[2/4] Loading Nsight Compute data...")
        if self.nsight_csv is not None:
            if not self.nsight_csv.is_file():
                raise FileNotFoundError(f"Nsight CSV not found: {self.nsight_csv}")
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

    def run_empirical_roofline_analysis(self) -> None:
        """Join real NCU DRAM bytes with the matching validated GEMM benchmark rows."""
        self.empirical_roofline_rows = []
        if not self.nsight_report or not self.benchmark_artifact:
            print("  Empirical arithmetic intensity unavailable: both a real NCU CSV "
                  "and a validated CUDA benchmark artifact are required.")
            return

        meta = self.benchmark_artifact["metadata"]
        shape = meta.get("shape") or {
            "M": meta["matrix_size"], "N": meta["matrix_size"], "K": meta["matrix_size"]
        }
        m, n, k = shape["M"], shape["N"], shape["K"]
        gemm_flops = 2 * m * n * k

        kernel_symbols = {
            "K0": "matmul_naive",
            "K1": "matmul_tiled_shmem",
            "K2": "matmul_register_tiled",
            "K3": "matmul_vectorized",
            "K4": "matmul_wmma_tensor_core",
        }
        for benchmark_row in self.kernel_benchmarks:
            symbol = kernel_symbols.get(benchmark_row["kernel_id"])
            if not symbol:
                continue
            ncu_row = next(
                (item for item in self.nsight_report.kernels
                 if symbol in item.kernel_name.lower()),
                None,
            )
            if ncu_row is None:
                continue
            required = {"dram_bytes_read", "dram_bytes_write"}
            if not required.issubset(set(ncu_row.available_metrics)):
                continue

            read_bytes = ncu_row.dram_bytes_read
            write_bytes = ncu_row.dram_bytes_write
            total_bytes = read_bytes + write_bytes
            if total_bytes <= 0:
                continue

            self.empirical_roofline_rows.append({
                "kernel_id": benchmark_row["kernel_id"],
                "kernel": benchmark_row["name"],
                "dram_bytes_read": read_bytes,
                "dram_bytes_write": write_bytes,
                "dram_bytes_total": total_bytes,
                "gemm_flops": gemm_flops,
                "arithmetic_intensity_flop_per_byte": gemm_flops / total_bytes,
                "benchmark_latency_ms": benchmark_row["latency_ms"],
                "benchmark_gflops": benchmark_row["gflops"],
                "input_precision": benchmark_row["input_precision"],
                "accumulation_precision": benchmark_row["accumulation_precision"],
                "output_precision": benchmark_row["output_precision"],
            })

        if self.empirical_roofline_rows:
            print("  Derived measured DRAM arithmetic intensity for "
                  f"{len(self.empirical_roofline_rows)} GEMM kernels.")
        else:
            print("  No matched kernels had both DRAM read and write counters; "
                  "the report will not invent arithmetic-intensity values.")

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
                f"- Shape: {meta.get('shape', {'M': meta.get('matrix_size'), 'N': meta.get('matrix_size'), 'K': meta.get('matrix_size')})}",
                f"- Layout: {meta.get('layout', 'not recorded')}",
                f"- FP32-to-FP16 input conversion: {meta.get('fp16_input_conversion_ms', 'not applicable')} ms",
                f"- Captured (UTC): {meta.get('captured_at_utc', 'timestamp not recorded')}",
                f"- Warm-up iterations: {meta.get('warmup_iterations', 'not recorded')}",
                f"- Timed iterations: {meta.get('timed_iterations', 'not recorded')}",
                f"- Reference: {ref.get('name', 'unknown')} "
                f"({ref['input_precision']} input / {ref['accumulation_precision']} accumulate / "
                f"{ref['output_precision']} output) — {ref['gflops']:.2f} GFLOPS "
                f"at {ref['latency_ms']:.4f} ms",
                "",
                "| Kernel | Input | Accumulator | Output | Latency (ms) | GFLOPS | % reference | Correctness |",
                "|---|---|---|---|---:|---:|---:|---|",
            ])
            for row in self.kernel_benchmarks:
                validation = row.get("validation") or {}
                state = ("PASS" if validation.get("passed") is True else
                         "FAIL" if validation.get("passed") is False else "not reported")
                pct = row.get("pct_of_cublas", row.get("pct_of_reference", ""))
                pct_text = f"{pct:.2f}%" if isinstance(pct, (int, float)) else "n/a"
                lines.append(
                    f"| {row['name']} | {row['input_precision']} | "
                    f"{row['accumulation_precision']} | {row['output_precision']} | "
                    f"{row['latency_ms']:.4f} | "
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

        def format_ncu_metric(metric, field_name: str) -> str:
            if metric.available_metrics and field_name not in metric.available_metrics:
                return "n/a (not collected)"
            return f"{getattr(metric, field_name):.2f}"

        if self.nsight_report:
            if "demo" in self.nsight_report.gpu_name.lower():
                lines.append("**Synthetic demo data — not a hardware profile.**")
            else:
                lines.append(f"Source CSV: {self.nsight_report.raw_csv_path or 'not recorded'}")
            lines.extend([
                "",
                "| Kernel | SM throughput (%) | Occupancy (%) | L1 hit (%) | L2 hit (%) | Tensor Core (%) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
            for metric in self.nsight_report.kernels:
                lines.append(
                    f"| {metric.kernel_name} | {format_ncu_metric(metric, 'sm_throughput_pct')} | "
                    f"{format_ncu_metric(metric, 'achieved_occupancy_pct')} | "
                    f"{format_ncu_metric(metric, 'l1_hit_rate_pct')} | "
                    f"{format_ncu_metric(metric, 'l2_hit_rate_pct')} | "
                    f"{format_ncu_metric(metric, 'tensor_core_pct')} |"
                )
        else:
            lines.append(
                "No real Nsight Compute CSV was supplied. Capture one with "
                "scripts/profile_ncu.sh; no profiler metrics are inferred."
            )

        if self.empirical_roofline_rows:
            lines.extend([
                "",
                "### Measured GEMM DRAM traffic and arithmetic intensity",
                "",
                "| Kernel | Input / accumulation | DRAM read (MiB) | DRAM write (MiB) | Total DRAM (MiB) | Arithmetic intensity (FLOP/byte) | Benchmark GFLOP/s |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
            for row in self.empirical_roofline_rows:
                lines.append(
                    f"| {row['kernel_id']} — {row['kernel']} | "
                    f"{row['input_precision']} / {row['accumulation_precision']} | "
                    f"{row['dram_bytes_read'] / (1024 ** 2):.2f} | "
                    f"{row['dram_bytes_write'] / (1024 ** 2):.2f} | "
                    f"{row['dram_bytes_total'] / (1024 ** 2):.2f} | "
                    f"{row['arithmetic_intensity_flop_per_byte']:.3f} | "
                    f"{row['benchmark_gflops']:.2f} |"
                )
            lines.extend([
                "",
                "Method: GEMM work is counted as 2×M×N×K FLOPs; DRAM read/write bytes "
                "come from the matched Nsight Compute kernel counters. GFLOP/s and "
                "latency come from the separate CUDA-event benchmark artifact. "
                "The arithmetic-intensity column is therefore derived from measured "
                "DRAM traffic, while throughput retains its benchmark-run provenance.",
                "",
                "Caveat: DRAM counters are observed at the device memory controllers and "
                "can include traffic from other GPU clients. Profile on an otherwise "
                "idle GPU and inspect the native .ncu-rep before drawing conclusions.",
            ])

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
        self.run_empirical_roofline_analysis()

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
            "empirical_roofline_rows": self.empirical_roofline_rows,
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
