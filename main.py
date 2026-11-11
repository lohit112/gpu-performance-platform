"""CLI entry point for the GPU Performance Engineering Platform.

Live CUDA measurements are executed on the target GPU. Roofline examples and
LLM throughput estimates are analytical; Nsight demo data is opt-in only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_roofline(args):
    from roofline.roofline_model import (
        HardwareRoof, build_demo_workloads, plot_roofline,
    )
    print("\nROOFLINE ANALYSIS — illustrative/demo workloads")
    roofs = [
        HardwareRoof.t4_spec(),
        HardwareRoof.a100_spec(),
        HardwareRoof.h100_spec(),
    ]
    output = Path(args.out_dir) / "roofline_analysis_demo.png"
    plot_roofline(roofs, build_demo_workloads(), str(output))
    print(f"Demo roofline chart: {output}")
    return output


def run_nsight(args):
    from profiling.nsight_profiler import NsightReport, visualize_nsight_report
    if args.nsight_csv:
        report = NsightReport.from_csv(args.nsight_csv)
        label = "REAL Nsight Compute CSV"
    elif args.demo_data:
        report = NsightReport._demo_report()
        label = "SYNTHETIC DEMO DATA — not a hardware profile"
    else:
        print("Nsight analysis skipped: provide --nsight-csv from a real ncu capture.")
        return None
    output = Path(args.out_dir) / "nsight_analysis.png"
    visualize_nsight_report(report, str(output))
    report.to_json(str(Path(args.out_dir) / "nsight_report.json"))
    print(f"{label}: {output}")
    return output


def run_llm(args):
    from llm_profiler.inference_profiler import (
        LLMInferenceProfiler, MODEL_CATALOG, GPU_CATALOG,
        visualize_inference_profile, print_profile_report,
    )
    model_key = args.model
    gpu_key = args.gpu
    if model_key not in MODEL_CATALOG:
        raise ValueError(f"Unknown model {model_key!r}. Options: {sorted(MODEL_CATALOG)}")
    if gpu_key not in GPU_CATALOG:
        raise ValueError(f"Unknown GPU {gpu_key!r}. Options: {sorted(GPU_CATALOG)}")
    profiler = LLMInferenceProfiler(mfu_estimate=args.mfu)
    profile = profiler.profile(model_key, gpu_key, context_len=args.context_len,
                               batch_size=args.batch_size, precision=args.precision)
    print_profile_report(profile)
    sweep = profiler.batch_profile(model_key, gpu_key, context_len=args.context_len,
                                   precision=args.precision)
    output = Path(args.out_dir) / "llm_inference_profile.png"
    visualize_inference_profile(sweep, str(output))
    print("Note: LLM throughput and latency values are analytical estimates, not live inference measurements.")
    return output


def run_kernels(args):
    from benchmark.runner import run_cuda_benchmark
    output = Path(args.benchmark_json)
    if not output.is_absolute():
        output = ROOT / output
    data = run_cuda_benchmark(
        ROOT, arch=args.arch, size=args.size,
        warmup=args.warmup, iterations=args.iterations,
        output_path=output,
    )
    reference = data["reference"]
    print(f"\nGPU: {data['metadata']['gpu_name']} (sm_{data['metadata']['compute_capability'].replace('.', '')})")
    print(f"cuBLAS reference: {reference['gflops']:.2f} GFLOPS")
    print(f"{'Kernel':<34} {'ms':>10} {'GFLOPS':>12} {'Validation':>12}")
    for row in data["results"]:
        validation = row.get("validation", {})
        state = "PASS" if validation.get("passed") else "NOT REPORTED"
        print(f"{row['name']:<34} {row['latency_ms']:>10.4f} {row['gflops']:>12.2f} {state:>12}")
    print(f"\nSaved real benchmark artifact: {output}")
    return output


def run_full_report(args):
    # The report generator consumes this run's artifact only. If CUDA fails,
    # do not accidentally load a stale latest.json from a previous session.
    benchmark_artifact = None
    try:
        benchmark_artifact = run_kernels(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\nLive CUDA benchmark not produced: {exc}")
        print("No synthetic or stale benchmark values will be substituted.")

    from report_gen.report_generator import ReportGenerator
    generator = ReportGenerator(
        output_dir=args.out_dir,
        author="Lohit Reddy",
        benchmark_json=str(benchmark_artifact) if benchmark_artifact is not None else None,
        nsight_csv=args.nsight_csv,
        include_demo_data=args.demo_data,
    )
    paths = generator.export_all(primary_gpu=args.gpu)
    print("\nGenerated report files:")
    for kind, path in paths.items():
        print(f"  {kind:10s}: {path}")
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="GPU Performance Engineering Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--module", choices=["roofline", "nsight", "llm", "kernels", "report"],
        default="report",
    )
    parser.add_argument("--model", default="llama_7b")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "int8", "fp8", "int4"],
                        default="fp16")
    parser.add_argument("--mfu", type=float, default=0.35)
    parser.add_argument("--arch", default="sm_75",
                        help="CUDA target architecture: sm_75, sm_80, sm_90, etc.")
    parser.add_argument("--size", type=int, default=4096,
                        help="Square GEMM dimension N (must be divisible by 16).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="CUDA warm-up launches per kernel.")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Timed CUDA launches per kernel.")
    parser.add_argument("--benchmark-json", default="results/benchmarks/latest.json")
    parser.add_argument("--nsight-csv", default=None,
                        help="CSV exported by a real Nsight Compute ncu capture")
    parser.add_argument("--demo-data", action="store_true",
                        help="Explicitly opt into synthetic demo workloads/profiler data")
    parser.add_argument("--out-dir", default="report_output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("GPU Performance Engineering Platform")
    print("=" * 48)

    try:
        if args.module == "roofline":
            run_roofline(args)
        elif args.module == "nsight":
            run_nsight(args)
        elif args.module == "llm":
            run_llm(args)
        elif args.module == "kernels":
            run_kernels(args)
        else:
            run_full_report(args)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
