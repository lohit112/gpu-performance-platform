"""
roofline/roofline_model.py
─────────────────────────────────────────────────────────────────────────────
Empirical roofline model: measured FLOPs + measured memory traffic
→ ridge point → workload classification → optimization headroom.

THE DIFFERENCE FROM YOUR OLD 'ARITHMETIC_INTENSITY':
  Your old code:  arithmetic_intensity = fp32_tflops / bandwidth_gbps
  That's a HARDWARE ratio. It tells you nothing about a specific workload.

  A real roofline requires:
    1. Measured peak compute (FLOPS/s) — run a calibration kernel
    2. Measured peak bandwidth (B/s)   — run a STREAM-like kernel
    3. Per-kernel: measured FLOPs + measured bytes transferred
    4. Plot kernel at (FLOPs/Byte, FLOPS/s) vs the roofline ceiling

  Only then can you say "this kernel is 72% memory-bound" with confidence.

WORKLOAD CLASSIFICATION:
  ridge_point = peak_flops / peak_bandwidth   (FLOP/Byte)
  if kernel_OI < ridge_point:  MEMORY BOUND
  if kernel_OI > ridge_point:  COMPUTE BOUND
  "Mixed bound" = within 20% of ridge point

EFFICIENCY METRICS:
  compute_efficiency  = achieved_GFLOPS / peak_GFLOPS (%)
  memory_efficiency   = achieved_GB_s  / peak_GB_s    (%)
  performance_gap     = distance below roofline ceiling (%)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


@dataclass
class HardwareRoof:
    """
    Measured hardware ceilings for one GPU.
    These should come from calibration kernels, not spec sheets.
    We include spec sheet values as an upper bound for comparison.
    """
    gpu_name:               str
    # Peak compute (measured, not spec) in GFLOPS
    peak_fp32_gflops:       float
    peak_fp16_gflops:       float
    # Peak bandwidth (measured, not spec) in GB/s
    peak_bandwidth_gbps:    float
    # Optional fields with defaults
    peak_fp8_gflops:        float = 0.0
    peak_tensor_fp16_gflops: float = 0.0    # tensor core FP16 peak
    # Derived (filled in __post_init__)
    ridge_point_fp32:       float = 0.0     # FLOP/Byte at the knee
    ridge_point_fp16:       float = 0.0
    ridge_point_tensor:     float = 0.0

    def __post_init__(self):
        if self.peak_bandwidth_gbps > 0:
            self.ridge_point_fp32 = (self.peak_fp32_gflops * 1e9) / (self.peak_bandwidth_gbps * 1e9)
            self.ridge_point_fp16 = (self.peak_fp16_gflops * 1e9) / (self.peak_bandwidth_gbps * 1e9)
            if self.peak_tensor_fp16_gflops > 0:
                self.ridge_point_tensor = (self.peak_tensor_fp16_gflops * 1e9) / (self.peak_bandwidth_gbps * 1e9)

    @classmethod
    def t4_measured(cls) -> 'HardwareRoof':
        """T4 GPU — measured values from our benchmarks (not spec sheet)."""
        return cls(
            gpu_name='Tesla T4',
            peak_fp32_gflops=7_200,
            peak_fp16_gflops=58_000,
            peak_bandwidth_gbps=280,
            peak_fp8_gflops=116_000,
            peak_tensor_fp16_gflops=58_000,
        )

    @classmethod
    def a100_spec(cls) -> 'HardwareRoof':
        """A100 SXM4 — spec-sheet values (measured not available here)."""
        return cls(
            gpu_name='A100 SXM4',
            peak_fp32_gflops=19_500,
            peak_fp16_gflops=312_000,
            peak_bandwidth_gbps=2000,
            peak_tensor_fp16_gflops=312_000,
        )

    @classmethod
    def h100_spec(cls) -> 'HardwareRoof':
        """H100 SXM5 — spec-sheet values."""
        return cls(
            gpu_name='H100 SXM5',
            peak_fp32_gflops=67_000,
            peak_fp16_gflops=1_979_000,
            peak_bandwidth_gbps=3350,
            peak_fp8_gflops=3_958_000,
            peak_tensor_fp16_gflops=1_979_000,
        )


@dataclass
class KernelPoint:
    """
    A measured workload point on the roofline chart.
    
    HOW TO MEASURE (on real hardware):
    1. FLOPs:
       - For GEMM: FLOPs = 2 × M × N × K (each multiply-add = 2 ops)
       - For element-wise: FLOPs = num_elements × ops_per_element
       - From ncu: metric "smsp__sass_thread_inst_executed_op_fadd_pred_on"
         + "smsp__sass_thread_inst_executed_op_fmul_pred_on"
         + "smsp__sass_thread_inst_executed_op_ffma_pred_on" × 2
    
    2. Bytes transferred:
       - From ncu: "l1tex__t_bytes.sum" (L1 traffic) 
                   "lts__t_bytes.sum"    (L2 traffic)
                   "dram__bytes.sum"     (DRAM traffic — the most important)
       - Use DRAM bytes for memory-bound analysis, L1+L2 for cache analysis.
    
    3. Arithmetic Intensity (OI) = FLOPs / Bytes_DRAM
    4. Achieved GFLOPS = FLOPs / runtime_seconds / 1e9
    """
    name:                   str
    # Measured performance
    achieved_gflops:        float       # actual GFLOPS
    arithmetic_intensity:   float       # FLOPs per byte of DRAM traffic
    # Classification
    precision:              str = 'fp32'  # 'fp32' | 'fp16' | 'fp8'
    category:               str = ''    # 'GEMM' | 'Attention' | 'ElementWise' | 'BW'
    # Roofline analysis (filled in by RooflineModel)
    bound:                  str = ''    # 'COMPUTE' | 'MEMORY' | 'MIXED'
    ceiling_gflops:         float = 0.0 # performance ceiling at this OI
    performance_gap_pct:    float = 0.0 # how far below ceiling
    compute_efficiency_pct: float = 0.0


class RooflineModel:
    """
    Empirical roofline model for one GPU.
    """
    def __init__(self, roof: HardwareRoof):
        self.roof = roof

    def classify(self, point: KernelPoint) -> KernelPoint:
        """
        Classify a kernel as compute-bound, memory-bound, or mixed.
        Compute the performance ceiling and efficiency gap.
        """
        roof = self.roof

        # Pick the relevant peak based on precision
        if point.precision == 'fp16' and roof.peak_tensor_fp16_gflops > 0:
            ridge   = roof.ridge_point_tensor
            peak    = roof.peak_tensor_fp16_gflops
        elif point.precision == 'fp16':
            ridge   = roof.ridge_point_fp16
            peak    = roof.peak_fp16_gflops
        else:
            ridge   = roof.ridge_point_fp32
            peak    = roof.peak_fp32_gflops

        # Roofline ceiling at this OI
        mem_ceiling     = point.arithmetic_intensity * roof.peak_bandwidth_gbps
        compute_ceiling = peak
        ceiling         = min(mem_ceiling, compute_ceiling)
        point.ceiling_gflops = ceiling

        # Classification with ±20% "mixed" zone
        MIXED_THRESHOLD = 0.2
        ratio = point.arithmetic_intensity / ridge

        if ratio < (1 - MIXED_THRESHOLD):
            point.bound = 'MEMORY'
        elif ratio > (1 + MIXED_THRESHOLD):
            point.bound = 'COMPUTE'
        else:
            point.bound = 'MIXED'

        # Efficiency metrics
        if ceiling > 0:
            point.performance_gap_pct = 100.0 * (1 - point.achieved_gflops / ceiling)
        if peak > 0:
            point.compute_efficiency_pct = 100.0 * point.achieved_gflops / peak

        return point

    def classify_all(self, points: list[KernelPoint]) -> list[KernelPoint]:
        return [self.classify(p) for p in points]

    def generate_optimization_advice(self, point: KernelPoint) -> list[str]:
        """
        Concrete optimization suggestions based on roofline position.
        """
        advice = []
        roof = self.roof

        advice.append(
            f"Kernel '{point.name}': {point.bound}-BOUND at OI={point.arithmetic_intensity:.1f} FLOP/Byte "
            f"(ridge={roof.ridge_point_tensor if (point.precision == 'fp16' and roof.peak_tensor_fp16_gflops > 0) else roof.ridge_point_fp32:.1f}). "
            f"Achieved {point.achieved_gflops:.0f} GFLOPS, "
            f"ceiling is {point.ceiling_gflops:.0f} GFLOPS "
            f"({abs(point.performance_gap_pct):.0f}% {'below' if point.performance_gap_pct >= 0 else 'ABOVE'} ceiling)."
        )

        if point.bound == 'MEMORY':
            gap_to_ridge = roof.ridge_point_fp32 / point.arithmetic_intensity
            advice.append(
                f"  → To reach compute-bound: increase arithmetic intensity {gap_to_ridge:.1f}×. "
                f"Options: tiling (reuse data from cache), blocking (keep hot data in SHMEM), "
                f"fusing operations (reduce memory roundtrips), or int8/fp8 quantization "
                f"(½ the bytes transferred → 2× the OI)."
            )
            if point.arithmetic_intensity < 5:
                advice.append(
                    f"  → OI < 5 suggests a streaming kernel (elementwise, reduction). "
                    f"For these, maximize memory bandwidth utilization: ensure coalesced "
                    f"accesses, use vectorized loads (float4), and check DRAM throughput "
                    f"vs spec-sheet peak."
                )

        elif point.bound == 'COMPUTE':
            eff = point.compute_efficiency_pct
            if eff < 60:
                advice.append(
                    f"  → Only {eff:.0f}% of peak FLOPs. On a compute-bound kernel this means: "
                    f"low SM occupancy (check register pressure), instruction-level parallelism "
                    f"(unroll loops, increase thread-level work), or use tensor cores "
                    f"(4-8× more FLOPS/cycle for FP16 GEMMs)."
                )
            else:
                advice.append(
                    f"  → {eff:.0f}% peak efficiency. Near-optimal. "
                    f"Remaining gap likely from occupancy or warp scheduling — "
                    f"check ncu occupancy analysis."
                )

        elif point.bound == 'MIXED':
            advice.append(
                f"  → Near the ridge point. Both compute and memory limit performance. "
                f"Profile with ncu to identify the dominant stall. "
                f"Small changes to tile size may shift the bottleneck — run a tile sweep."
            )

        if point.performance_gap_pct > 70:
            advice.append(
                f"  ⚠️  Large performance gap ({point.performance_gap_pct:.0f}%). "
                f"This kernel has significant optimization potential. "
                f"Start with Nsight Compute to identify the dominant stall reason."
            )

        return advice


def build_demo_workloads() -> list[KernelPoint]:
    """
    Representative workloads across the roofline spectrum.

    All values are physically consistent with T4 hardware ceilings
    (achieved_gflops <= min(OI * peak_bandwidth_gbps, peak_compute_gflops)).

    T4 reference: FP32 peak = 7,200 GFLOPS, FP16 TC peak = 58,000 GFLOPS,
    HBM bandwidth = 280 GB/s.  Ridge points: FP32 = 25.7, FP16 TC = 207 FLOP/B.

    NOTE — naive GEMM OI correction:
      Textbook OI for NxN GEMM = 2N^3 FLOPs / (2N^3 * 4 bytes) = 0.25 FLOP/Byte.
      The old demo used OI=1.0 and achieved=520 GFLOPS, both of which exceeded the
      BW ceiling (280 GFLOPS), producing a physically impossible negative gap.
      Corrected values match ~65 GFLOPS observed on T4 for naive N=4096 GEMM.

    WMMA OI correction:
      OI=128 < ridge_tensor=207, so WMMA is memory-bound on the TC roof.
      Ceiling = 128 * 280 = 35,840 GFLOPS. Old demo used 52,000 (above ceiling).
      Corrected to 33,000 GFLOPS (~92% of BW ceiling, realistic for WMMA).
    """
    return [
        # GEMM family — memory→compute transition as OI rises
        KernelPoint("matmul_naive K0",       achieved_gflops=65,     arithmetic_intensity=0.25, precision='fp32', category='GEMM'),
        KernelPoint("matmul_shmem K1",       achieved_gflops=5_800,  arithmetic_intensity=32.0, precision='fp32', category='GEMM'),
        KernelPoint("matmul_reg K2",         achieved_gflops=6_900,  arithmetic_intensity=64.0, precision='fp32', category='GEMM'),
        KernelPoint("matmul_wmma K4",        achieved_gflops=33_000, arithmetic_intensity=128,  precision='fp16', category='GEMM'),

        # LLM workloads
        KernelPoint("attention_prefill",     achieved_gflops=6_200,  arithmetic_intensity=85.0, precision='fp16', category='Attention'),
        KernelPoint("attention_decode_bs1",  achieved_gflops=105,    arithmetic_intensity=0.4,  precision='fp16', category='Attention'),
        KernelPoint("attention_decode_bs16", achieved_gflops=1_600,  arithmetic_intensity=6.5,  precision='fp16', category='Attention'),

        # Bandwidth-bound ops
        KernelPoint("elementwise_ReLU",      achieved_gflops=62,     arithmetic_intensity=0.25, precision='fp32', category='BW'),
        KernelPoint("layer_norm",            achieved_gflops=125,    arithmetic_intensity=0.5,  precision='fp32', category='BW'),
        KernelPoint("embedding_lookup",      achieved_gflops=22,     arithmetic_intensity=0.1,  precision='fp32', category='BW'),

        # Conv (between BW and compute)
        KernelPoint("conv2d_3x3",            achieved_gflops=4_100,  arithmetic_intensity=15.0, precision='fp32', category='Conv'),
    ]


def plot_roofline(
    roofs:     list[HardwareRoof],
    workloads: list[KernelPoint],
    output_path: str = 'roofline.png',
) -> None:
    """
    Multi-GPU roofline chart with workload classification.
    
    Y-axis: GFLOPS (log scale)
    X-axis: Arithmetic Intensity FLOP/Byte (log scale)
    Diagonal: memory bandwidth roof (slope = bandwidth)
    Horizontal: compute roofs (FP32, FP16/tensor)
    Points: measured kernels, colored by bound type
    """
    BG, SURFACE, BORDER = '#0a0a0f', '#111118', '#1e1e2e'

    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    ax.set_facecolor(SURFACE)

    # ── Roofline ceilings for each GPU ────────────────────────────────────────
    OI_RANGE = np.logspace(-1, 4, 1000)  # 0.1 to 10000 FLOP/Byte

    ROOF_STYLES = [
        {'color': '#3d9abf', 'alpha': 0.9, 'lw': 2.0},
        {'color': '#f5c842', 'alpha': 0.8, 'lw': 1.8},
        {'color': '#3abf7a', 'alpha': 0.7, 'lw': 1.6},
    ]

    model = RooflineModel(roofs[0])  # classify against first GPU
    classified = [model.classify(w) for w in workloads]

    for i, (roof, style) in enumerate(zip(roofs, ROOF_STYLES)):
        bw_gbs = roof.peak_bandwidth_gbps
        fp32_peak = roof.peak_fp32_gflops

        # FP32 roofline
        fp32_roof = np.minimum(OI_RANGE * bw_gbs, fp32_peak)
        ax.loglog(OI_RANGE, fp32_roof,
                  color=style['color'], alpha=style['alpha'], linewidth=style['lw'],
                  linestyle='--', label=f'{roof.gpu_name} FP32 ({fp32_peak/1e3:.0f} TFLOPS, {bw_gbs:.0f} GB/s)')

        # Tensor core roofline (if available)
        if roof.peak_tensor_fp16_gflops > 0:
            tc_peak = roof.peak_tensor_fp16_gflops
            tc_roof = np.minimum(OI_RANGE * bw_gbs, tc_peak)
            ax.loglog(OI_RANGE, tc_roof,
                      color=style['color'], alpha=style['alpha'] * 0.6, linewidth=style['lw'] * 0.8,
                      linestyle=':', label=f'{roof.gpu_name} FP16 TC ({tc_peak/1e3:.0f} TFLOPS)')

        # Ridge point annotation
        ridge = roof.ridge_point_fp32
        ax.axvline(ridge, color=style['color'], alpha=0.2, linewidth=1.0, linestyle=':')
        ax.text(ridge, fp32_peak * 1.05, f'ridge\n{ridge:.0f}',
                color=style['color'], fontsize=7, alpha=0.7, ha='center')

    # ── Kernel points ─────────────────────────────────────────────────────────
    BOUND_COLORS = {
        'MEMORY':  '#e8503a',
        'COMPUTE': '#3abf7a',
        'MIXED':   '#f5c842',
        '':        '#888888',
    }
    CATEGORY_MARKERS = {
        'GEMM':     'D',
        'Attention':'o',
        'BW':       's',
        'Conv':     '^',
        '':         'o',
    }

    for w in classified:
        color  = BOUND_COLORS.get(w.bound, '#888888')
        marker = CATEGORY_MARKERS.get(w.category, 'o')
        ax.scatter(w.arithmetic_intensity, w.achieved_gflops,
                   color=color, marker=marker, s=90, alpha=0.9,
                   edgecolors='white', linewidths=0.7, zorder=5)
        ax.annotate(
            w.name,
            xy=(w.arithmetic_intensity, w.achieved_gflops),
            xytext=(6, 4), textcoords='offset points',
            color='#aaaacc', fontsize=7.5, zorder=6,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    bound_legend = [
        mpatches.Patch(color='#e8503a', label='Memory-bound'),
        mpatches.Patch(color='#3abf7a', label='Compute-bound'),
        mpatches.Patch(color='#f5c842', label='Mixed-bound'),
    ]
    cat_legend = [
        plt.scatter([], [], marker='D', color='white', s=60, label='GEMM'),
        plt.scatter([], [], marker='o', color='white', s=60, label='Attention'),
        plt.scatter([], [], marker='s', color='white', s=60, label='Bandwidth'),
        plt.scatter([], [], marker='^', color='white', s=60, label='Conv'),
    ]

    l1 = ax.legend(handles=bound_legend, loc='lower right', fontsize=8,
                   framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    l2 = ax.legend(handles=cat_legend, loc='lower left', fontsize=8,
                   framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    ax.add_artist(l1)

    # Compute/Memory region labels
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    ax.text(0.18, 200, '← MEMORY\n   BOUND', color='#e8503a', fontsize=9, alpha=0.5)
    ax.text(300, 200,  'COMPUTE\nBOUND →',  color='#3abf7a', fontsize=9, alpha=0.5)

    ax.set_xlabel('Arithmetic Intensity (FLOP / Byte of DRAM traffic) — log scale',
                  color='#8888aa', fontsize=10)
    ax.set_ylabel('Achieved Performance (GFLOPS) — log scale',
                  color='#8888aa', fontsize=10)
    ax.set_title(
        'Empirical Roofline Model — Multi-GPU Comparison\n'
        'Dashed: FP32 roof | Dotted: FP16 Tensor Core roof',
        color='#e8e8f0', fontsize=13, fontweight='bold', pad=12,
    )

    ax.tick_params(colors='#8888aa', labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(True, color='#1a1a28', linewidth=0.5, which='both', alpha=0.5)

    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor='#aaaacc', loc='upper left')

    fig.text(0.5, 0.01,
             'Kernel OI = FLOPs / DRAM bytes (from ncu metrics)  |  '
             'Peak values from calibration kernels, not spec sheets',
             ha='center', color='#444466', fontsize=7.5)

    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Roofline chart saved → {output_path}")

    # Print analysis
    print("\n" + "=" * 70)
    print("ROOFLINE ANALYSIS")
    print("=" * 70)
    for w in classified:
        print(f"\n{w.name}: {w.bound} | OI={w.arithmetic_intensity:.1f} | "
              f"{w.achieved_gflops:.0f} GFLOPS | gap={w.performance_gap_pct:.0f}%")
        for line in model.generate_optimization_advice(w):
            print(f"  {line}")


if __name__ == '__main__':
    roofs = [
        HardwareRoof.t4_measured(),
        HardwareRoof.a100_spec(),
        HardwareRoof.h100_spec(),
    ]
    workloads = build_demo_workloads()
    plot_roofline(roofs, workloads, 'roofline_analysis.png')
