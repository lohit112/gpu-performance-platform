"""
profiling/nsight_profiler.py
─────────────────────────────────────────────────────────────────────────────
Nsight Compute integration: parse ncu reports and extract SM utilization,
occupancy, warp stall reasons, cache hit rates, and tensor core utilization.

WHY THIS IS THE MOST IMPORTANT MODULE IN THE UPGRADE:
  ChatGPT's suggestion to "add Nsight" was correct but vague.
  The real engineering work is:
  1. Knowing WHICH metrics to collect (not all 400+ ncu metrics matter)
  2. Knowing HOW to interpret them (stall reasons, occupancy limiters)
  3. Knowing HOW to generate ACTIONABLE recommendations from the numbers

  The module provides a CSV adapter and rule-based analysis. Synthetic demo
  fixtures are explicitly marked and are not substitutes for real profiling.

HOW TO RUN:
  # Step 1: Profile your binary with ncu
  ncu --metrics <available-metric-list> -o matmul_profile.ncu-rep ./matmul_bench
  ncu --import matmul_profile.ncu-rep --csv --page raw > ncu_raw.csv

  # Step 2: Pass the CSV to this module
  from profiling.nsight_profiler import NsightReport, visualize_nsight_report
  report = NsightReport.from_csv('ncu_raw.csv')
  visualize_nsight_report(report)

METRICS GLOSSARY (the 12 that matter most):
  sm__throughput                  → SM compute utilization (%)
  sm__warps_active                → Achieved occupancy (%)
  l1tex__t_sector_hit_rate        → L1/tex cache hit rate (%)
  lts__t_sector_hit_rate          → L2 cache hit rate (%)
  stall_mio_throttle              → % cycles stalled on memory I/O queue
  stall_long_scoreboard           → % cycles waiting for L2/HBM (cache miss)
  stall_not_selected              → % warps not selected despite being ready
  stall_barrier                   → % cycles at __syncthreads() waiting
  stall_math_throttle             → % cycles stalled on FP unit full pipeline
  pipe_tensor_cycles_active       → Tensor core utilization (%)
  dram_throughput                 → % of peak HBM bandwidth used
  shared_efficiency               → Shared memory efficiency (% of ideal)
"""

from __future__ import annotations

import re
import csv
import json
import subprocess
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


@dataclass
class NsightMetrics:
    """
    Core subset of Nsight Compute metrics.
    Units are percentages (% of peak) unless noted.
    
    We focus on 12 metrics because they cover the 5 bottleneck categories:
    Compute, Memory, Occupancy, Latency, and Tensor Cores.
    """
    kernel_name:            str   = "unknown"

    # ── Compute utilization ──────────────────────────────────────────────────
    sm_throughput_pct:      float = 0.0   # SM compute utilization vs peak
    tensor_core_pct:        float = 0.0   # Tensor core pipeline utilization

    # ── Occupancy ────────────────────────────────────────────────────────────
    achieved_occupancy_pct: float = 0.0   # active warps / max warps per SM
    theoretical_occupancy:  float = 0.0   # occupancy limited by resources
    # Occupancy limiter (what's blocking higher occupancy):
    # 'registers' | 'shared_memory' | 'block_size' | 'waves'
    occupancy_limiter:      str   = "unknown"

    # ── Warp stall breakdown (% of warp cycles stalled for each reason) ──────
    stall_mio_throttle:     float = 0.0   # Memory I/O queue full
    stall_long_scoreboard:  float = 0.0   # Waiting for L2/HBM (cache miss)
    stall_not_selected:     float = 0.0   # Warp eligible but not selected (good)
    stall_barrier:          float = 0.0   # Waiting at __syncthreads()
    stall_math_throttle:    float = 0.0   # FP pipeline full (good problem)
    stall_short_scoreboard: float = 0.0   # Waiting for register writeback

    # ── Memory hierarchy hit rates ───────────────────────────────────────────
    l1_hit_rate_pct:        float = 0.0   # L1 texture/load cache hit rate
    l2_hit_rate_pct:        float = 0.0   # L2 unified cache hit rate
    dram_throughput_pct:    float = 0.0   # % of peak HBM bandwidth achieved

    # ── Shared memory ────────────────────────────────────────────────────────
    shmem_efficiency_pct:   float = 0.0   # efficiency = useful / total SHMEM ops
    shmem_bank_conflicts:   float = 0.0   # bank conflicts per warp instruction

    # ── Registers ────────────────────────────────────────────────────────────
    registers_per_thread:   int   = 0     # register usage (too high → low occ.)
    available_metrics:      list[str] = field(default_factory=list)


@dataclass
class NsightReport:
    """
    Complete Nsight Compute report: one or more kernel profiles.
    """
    gpu_name:       str
    cuda_version:   str
    kernels:        list[NsightMetrics] = field(default_factory=list)
    raw_csv_path:   Optional[str] = None

    @classmethod
    def from_csv(cls, csv_path: str) -> 'NsightReport':
        """
        Parse ncu --csv output into structured NsightReport.
        
        ncu CSV format:
          "ID","Process ID","Process Name","Host Name","Kernel Name",
          "Kernel Time","Context","Stream","Section Name","Metric Name",
          "Metric Unit","Metric Value"
        """
        kernels: dict[str, NsightMetrics] = {}
        gpu_name = "Unknown GPU"
        cuda_ver = "Unknown"

        # Metric name → NsightMetrics field mapping
        METRIC_MAP = {
            'sm__throughput.avg.pct_of_peak_sustained_elapsed':
                'sm_throughput_pct',
            'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed':
                'tensor_core_pct',
            'sm__warps_active.avg.pct_of_peak_sustained_active':
                'achieved_occupancy_pct',
            'l1tex__t_sector_hit_rate.pct':
                'l1_hit_rate_pct',
            'lts__t_sector_hit_rate.pct':
                'l2_hit_rate_pct',
            'dram__throughput.avg.pct_of_peak_sustained_elapsed':
                'dram_throughput_pct',
            'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct':
                'stall_mio_throttle',
            'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct':
                'stall_long_scoreboard',
            'smsp__warp_issue_stalled_not_selected_per_warp_active.pct':
                'stall_not_selected',
            'smsp__warp_issue_stalled_barrier_per_warp_active.pct':
                'stall_barrier',
            'smsp__warp_issue_stalled_math_throttle_per_warp_active.pct':
                'stall_math_throttle',
            'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct':
                'stall_short_scoreboard',
            'l1tex__data_pipe_lsu_wavefronts_mem_shared_bank_conflict.sum':
                'shmem_bank_conflicts',
        }

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel_name = row.get('Kernel Name', 'unknown').strip()
                    if not kernel_name:
                        continue

                    if kernel_name not in kernels:
                        kernels[kernel_name] = NsightMetrics(kernel_name=kernel_name)

                    # Nsight CSV has two common layouts:
                    #   (1) section/detail CSV: one row per metric, Metric Name/Value columns
                    #   (2) raw page CSV: one row per kernel, metric names are CSV columns
                    metric_name = row.get('Metric Name', '').strip()
                    if metric_name:
                        candidates = [(metric_name, row.get('Metric Value', ''))]
                    else:
                        candidates = [(name, row.get(name, '')) for name in METRIC_MAP]

                    for name, raw_value in candidates:
                        if name not in METRIC_MAP or raw_value is None:
                            continue
                        metric_val = str(raw_value).strip().replace(',', '')
                        if not metric_val or metric_val in {'-', 'N/A', 'n/a'}:
                            continue
                        try:
                            val = float(metric_val)
                            field_name = METRIC_MAP[name]
                            setattr(kernels[kernel_name], field_name, val)
                            if field_name not in kernels[kernel_name].available_metrics:
                                kernels[kernel_name].available_metrics.append(field_name)
                        except (ValueError, TypeError):
                            continue

        except FileNotFoundError:
            raise FileNotFoundError(f"Nsight Compute CSV not found: {csv_path}")

        if not kernels:
            raise ValueError(
                f"No kernel rows were parsed from {csv_path}. "
                "Check that this is an ncu --csv export."
            )
        mapped_count = 0
        for metric in kernels.values():
            mapped_count += sum(
                getattr(metric, field_name) != 0
                for field_name in set(METRIC_MAP.values())
            )
        if mapped_count == 0:
            raise ValueError(
                f"Kernel rows were found in {csv_path}, but none of the configured "
                "metric columns matched. Check metric names against this GPU/Nsight version."
            )

        return cls(
            gpu_name=gpu_name,
            cuda_version=cuda_ver,
            kernels=list(kernels.values()),
            raw_csv_path=csv_path,
        )

    @classmethod
    def _demo_report(cls) -> 'NsightReport':
        """
        Synthetic placeholder values used only to exercise charts and report layouts.

        These values are not Nsight captures and must never be cited as measurements.
        Use this method only after explicitly selecting demo mode.
        """
        return cls(
            gpu_name="Tesla T4 (demo data)",
            cuda_version="12.1",
            kernels=[
                NsightMetrics(
                    kernel_name="matmul_naive",
                    sm_throughput_pct=15.2,        # only 15% SM utilization
                    tensor_core_pct=0.0,           # no tensor cores used
                    achieved_occupancy_pct=87.5,   # high occupancy...
                    stall_long_scoreboard=72.1,    # ...but mostly stalled on HBM
                    stall_mio_throttle=8.3,
                    stall_not_selected=12.5,
                    l1_hit_rate_pct=12.3,          # terrible cache hit rate
                    l2_hit_rate_pct=31.5,
                    dram_throughput_pct=89.6,      # near max HBM BW — it's BW-bound
                    shmem_efficiency_pct=0.0,
                    registers_per_thread=16,
                ),
                NsightMetrics(
                    kernel_name="matmul_tiled_shmem",
                    sm_throughput_pct=67.8,        # much better SM utilization
                    tensor_core_pct=0.0,
                    achieved_occupancy_pct=62.5,   # lower occ. due to SHMEM usage
                    stall_long_scoreboard=18.2,    # cache misses reduced 4x
                    stall_barrier=31.4,            # __syncthreads() visible now
                    stall_not_selected=22.1,
                    l1_hit_rate_pct=68.4,          # SHMEM acting as L0 cache
                    l2_hit_rate_pct=84.2,          # great L2 hit rate with tiling
                    dram_throughput_pct=24.3,
                    shmem_efficiency_pct=98.2,     # no bank conflicts
                    shmem_bank_conflicts=0.0,
                    registers_per_thread=24,
                ),
                NsightMetrics(
                    kernel_name="matmul_register_tiled",
                    sm_throughput_pct=81.4,
                    tensor_core_pct=0.0,
                    achieved_occupancy_pct=37.5,   # low occ. — many registers
                    stall_long_scoreboard=8.1,
                    stall_math_throttle=42.3,      # stalled on FP pipeline (good!)
                    stall_not_selected=28.9,
                    l1_hit_rate_pct=82.1,
                    l2_hit_rate_pct=91.5,
                    dram_throughput_pct=12.1,
                    shmem_efficiency_pct=97.8,
                    registers_per_thread=64,       # many registers = low occupancy
                ),
                NsightMetrics(
                    kernel_name="matmul_wmma_tensor_core",
                    sm_throughput_pct=92.3,
                    tensor_core_pct=88.7,          # tensor cores active!
                    achieved_occupancy_pct=50.0,
                    stall_math_throttle=55.1,      # compute-bound (expected)
                    stall_long_scoreboard=6.2,
                    stall_not_selected=21.4,
                    l1_hit_rate_pct=79.3,
                    l2_hit_rate_pct=89.8,
                    dram_throughput_pct=15.7,
                    shmem_efficiency_pct=96.1,
                    registers_per_thread=48,
                ),
            ]
        )

    def generate_recommendations(self, kernel: NsightMetrics) -> list[str]:
        """
        Rule-based optimization recommendations from Nsight metrics.
        
        This is the core engineering insight: knowing which metric → which fix.
        These rules are from NVIDIA's own optimization guides and workshop slides.
        """
        recs = []

        # In a real CSV, analyze only counters that were actually collected.
        # Empty availability metadata is reserved for the built-in demo fixtures.
        def observed(field_name: str) -> bool:
            return not kernel.available_metrics or field_name in kernel.available_metrics

        # Memory-bandwidth bound: high stall_long_scoreboard, high DRAM throughput
        if observed('stall_long_scoreboard') and kernel.stall_long_scoreboard > 40:
            recs.append(
                f"Memory bound: {kernel.stall_long_scoreboard:.0f}% warp cycles stalled "
                f"waiting for L2/HBM. Fix: increase cache hit rate with tiling, "
                f"or use prefetching (__ldg() for read-only data)."
            )

        # Low compute utilization
        if observed('sm_throughput_pct') and kernel.sm_throughput_pct < 50:
            recs.append(
                f"Low SM utilization: {kernel.sm_throughput_pct:.0f}% of peak. "
                f"SM is underloaded. Fix: increase tile size, increase batch size, "
                f"or launch more thread blocks."
            )

        # Barrier stalls — too many __syncthreads() or load imbalance
        if observed('stall_barrier') and kernel.stall_barrier > 25:
            recs.append(
                f"Barrier stalls: {kernel.stall_barrier:.0f}% cycles at __syncthreads(). "
                f"Consider double-buffering (async pipeline) to overlap sync with compute: "
                f"use __pipeline_commit/__pipeline_wait (Ampere+) or manual double-buffering."
            )

        # Low L1 hit rate
        if observed('l1_hit_rate_pct') and kernel.l1_hit_rate_pct < 30:
            recs.append(
                f"Poor L1 cache reuse: {kernel.l1_hit_rate_pct:.0f}% hit rate. "
                f"Access pattern is not cache-friendly. Fix: use shared memory for "
                f"reused data (K1 pattern), or __ldg() for read-once global data."
            )

        # Bank conflicts in shared memory
        if observed('shmem_bank_conflicts') and kernel.shmem_bank_conflicts > 1000:
            recs.append(
                f"Shared memory bank conflicts: {kernel.shmem_bank_conflicts:.0f} total. "
                f"Fix: pad SHMEM arrays by +1 column: float sA[TILE][TILE+1]. "
                f"This spreads accesses across different banks."
            )

        # Low occupancy with high register pressure
        if (observed('achieved_occupancy_pct') and observed('registers_per_thread')
                and kernel.achieved_occupancy_pct < 40 and kernel.registers_per_thread > 48):
            recs.append(
                f"Low occupancy (register pressure): {kernel.registers_per_thread} "
                f"regs/thread limits active warps to {kernel.achieved_occupancy_pct:.0f}%. "
                f"Fix: add __launch_bounds__({{}}) to cap register allocation. "
                f"Trade-off: spills to local memory may appear — check with ncu --metrics "
                f"l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
            )

        # Tensor cores not used when they should be
        if observed('tensor_core_pct') and kernel.tensor_core_pct < 10 and 'wmma' not in kernel.kernel_name.lower():
            recs.append(
                f"🟢 OPPORTUNITY: Tensor core utilization is {kernel.tensor_core_pct:.0f}%. "
                f"Convert this kernel to use WMMA (Volta+) or PTX-level tensor core ops "
                f"for 4-8× more TFLOPS on FP16 workloads."
            )

        # Math throttle stalls — this is actually GOOD (compute-bound)
        if observed('stall_math_throttle') and kernel.stall_math_throttle > 40:
            recs.append(
                f"Compute-bound: {kernel.stall_math_throttle:.0f}% cycles stalled on "
                f"FP pipeline full. This is the GOOD kind of stall — math units are busy. "
                f"Further optimization requires increasing instruction-level parallelism or "
                f"switching to tensor cores."
            )

        if not recs:
            recs.append("No major bottlenecks detected. Kernel appears reasonably well-optimized.")

        return recs

    def to_json(self, path: str) -> None:
        data = {
            'gpu_name':    self.gpu_name,
            'cuda_version': self.cuda_version,
            'kernels':     [asdict(k) for k in self.kernels],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Nsight report saved → {path}")


def visualize_nsight_report(
    report: NsightReport,
    output_path: str = 'nsight_analysis.png',
) -> None:
    """
    3-panel Nsight visualization:
    Panel A: SM utilization + Tensor core utilization per kernel
    Panel B: Warp stall breakdown (stacked horizontal bar)
    Panel C: Memory hierarchy hit rates

    This is the chart that separates GPU engineers from GPU users.
    """
    if not report.kernels:
        print("No kernels to visualize.")
        return

    BG, SURFACE, BORDER, GRID = '#0a0a0f', '#111118', '#1e1e2e', '#1a1a28'

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), facecolor=BG)
    kernels = report.kernels
    knames  = [k.kernel_name.replace('matmul_', '').replace('_', '\n') for k in kernels]
    n       = len(kernels)
    x       = np.arange(n)
    width   = 0.35

    def metric_value(kernel, field_name):
        # A value absent from the capture is not a measured zero.
        if kernel.available_metrics and field_name not in kernel.available_metrics:
            return np.nan
        return getattr(kernel, field_name)

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors='#8888aa', labelsize=8)
        ax.xaxis.label.set_color('#8888aa')
        ax.yaxis.label.set_color('#8888aa')
        ax.title.set_color('#e8e8f0')
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=0.5)

    # ── Panel A: SM + Tensor Core utilization ─────────────────────────────────
    ax = axes[0]
    sm_values = [metric_value(k, 'sm_throughput_pct') for k in kernels]
    tc_values = [metric_value(k, 'tensor_core_pct') for k in kernels]
    ax.bar(x - width/2, sm_values, width, color='#3d9abf', alpha=0.85, label='SM Util %', zorder=3)
    ax.bar(x + width/2, tc_values, width, color='#f5c842', alpha=0.85, label='Tensor Core %', zorder=3)

    for i, k in enumerate(kernels):
        sm_value = sm_values[i]
        tc_value = tc_values[i]
        if np.isfinite(sm_value):
            ax.text(i - width/2, sm_value + 1.5, f'{sm_value:.0f}%',
                    ha='center', va='bottom', color='#3d9abf', fontsize=7, fontweight='bold')
        if np.isfinite(tc_value) and tc_value > 1:
            ax.text(i + width/2, tc_value + 1.5, f'{tc_value:.0f}%',
                    ha='center', va='bottom', color='#f5c842', fontsize=7, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(knames, color='#8888aa', fontsize=8)
    ax.set_ylim(0, 110); ax.set_ylabel('% of Peak', fontsize=9)
    ax.set_title('A  Compute Utilization\n(SM + Tensor Core)', fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#8888aa')
    style(ax)

    # ── Panel B: Warp stall breakdown ─────────────────────────────────────────
    ax = axes[1]
    stall_fields = [
        'stall_long_scoreboard', 'stall_mio_throttle', 'stall_barrier',
        'stall_math_throttle', 'stall_not_selected',
    ]
    stalls_complete = all(
        not k.available_metrics or all(field in k.available_metrics for field in stall_fields)
        for k in kernels
    )
    if stalls_complete:
        STALL_COLORS = {
            'Long Scoreboard\n(HBM miss)': ('#e8503a', [k.stall_long_scoreboard for k in kernels]),
            'MIO Throttle\n(mem queue)':   ('#f5c842', [k.stall_mio_throttle for k in kernels]),
            'Barrier\n(syncthreads)':      ('#bf8c3d', [k.stall_barrier for k in kernels]),
            'Math Throttle\n(FP full)':    ('#3abf7a', [k.stall_math_throttle for k in kernels]),
            'Not Selected\n(eligible)':   ('#4a4a6a', [k.stall_not_selected for k in kernels]),
        }
        bottoms = np.zeros(n)
        for label, (color, vals) in STALL_COLORS.items():
            vals = np.array(vals)
            ax.bar(x, vals, width=0.6, bottom=bottoms, color=color, alpha=0.85, label=label, zorder=3)
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 5:
                    ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                            color='white', fontsize=7, fontweight='bold')
            bottoms += vals
    else:
        ax.text(0.5, 0.5, "Warp-stall counters were not collected\nin this Nsight capture.",
                transform=ax.transAxes, ha='center', va='center',
                color='#aaaacc', fontsize=10)
        ax.set_xlim(-0.5, max(0.5, n - 0.5))

    ax.set_xticks(x); ax.set_xticklabels(knames, color='#8888aa', fontsize=8)
    ax.set_ylabel('% of Warp Cycles', fontsize=9)
    ax.set_title('B  Warp Stall Analysis\n(bottleneck diagnosis)', fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=7, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#8888aa',
              loc='upper right', ncol=1)
    style(ax)
    ax.grid(False)

    # ── Panel C: Memory hierarchy hit rates ───────────────────────────────────
    ax = axes[2]
    l1_values = [metric_value(k, 'l1_hit_rate_pct') for k in kernels]
    l2_values = [metric_value(k, 'l2_hit_rate_pct') for k in kernels]
    dram_values = [metric_value(k, 'dram_throughput_pct') for k in kernels]
    ax.bar(x - width/2, l1_values, width, color='#3d6abf', alpha=0.85, label='L1 Hit Rate %', zorder=3)
    ax.bar(x + width/2, l2_values, width, color='#76b900', alpha=0.85, label='L2 Hit Rate %', zorder=3)

    # Overlay only measured DRAM throughput values.
    ax2_twin = ax.twinx()
    ax2_twin.plot(x, dram_values, 'o--', color='#e8503a', linewidth=2,
                  markersize=7, label='DRAM BW %', zorder=5)
    ax2_twin.set_ylabel('DRAM BW % of Peak', color='#e8503a', fontsize=9)
    ax2_twin.tick_params(colors='#e8503a')
    ax2_twin.set_ylim(0, 110)
    for sp in ax2_twin.spines.values(): sp.set_edgecolor(BORDER)

    for i in range(n):
        if np.isfinite(l1_values[i]):
            ax.text(i - width/2, l1_values[i] + 1.5, f'{l1_values[i]:.0f}%',
                    ha='center', va='bottom', color='#3d6abf', fontsize=7, fontweight='bold')
        if np.isfinite(l2_values[i]):
            ax.text(i + width/2, l2_values[i] + 1.5, f'{l2_values[i]:.0f}%',
                    ha='center', va='bottom', color='#76b900', fontsize=7, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(knames, color='#8888aa', fontsize=8)
    ax.set_ylim(0, 110); ax.set_ylabel('Cache Hit Rate %', fontsize=9)
    ax.set_title('C  Memory Hierarchy\n(L1, L2, DRAM)', fontsize=10, fontweight='bold', pad=8)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, framealpha=0.3,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor='#8888aa')
    style(ax)

    fig.suptitle(
        f'Nsight Compute Analysis — {report.gpu_name}',
        color='#e8e8f0', fontsize=15, fontweight='bold', y=0.98,
    )

    # Print recommendations
    print("\n" + "=" * 70)
    print("NSIGHT OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    for k in report.kernels:
        print(f"\nKernel: {k.kernel_name}")
        print("-" * 40)
        recs = report.generate_recommendations(k)
        for r in recs:
            print(f"  {r}")

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"\nNsight visualization saved → {output_path}")


def run_ncu_profiling(
    binary_path: str,
    output_csv: str = "ncu_raw.csv",
    metrics: list[str] | None = None,
) -> str:
    """
    Capture an Nsight Compute report, then export a clean raw-page CSV.

    The native .ncu-rep is kept beside the CSV for inspection in ncu-ui.
    Metric availability depends on GPU architecture and installed NCU version.
    """
    from pathlib import Path

    if metrics is None:
        metrics = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "smsp__warp_issue_stalled_math_throttle_per_warp_active.pct",
        ]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".ncu-rep")

    command = [
        "ncu",
        "--metrics", ",".join(metrics),
        "-o", str(report_path),
        binary_path,
    ]
    print(f"Running: {' '.join(command)}")
    try:
        profile = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if profile.returncode != 0:
            raise RuntimeError(
                f"ncu profiling failed ({profile.returncode}): "
                f"{profile.stderr.strip() or profile.stdout.strip()}"
            )

        export = subprocess.run(
            ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
            capture_output=True, text=True, timeout=300,
        )
        if export.returncode != 0:
            raise RuntimeError(
                f"ncu CSV export failed ({export.returncode}): "
                f"{export.stderr.strip() or export.stdout.strip()}"
            )
        output_path.write_text(export.stdout, encoding="utf-8")
        print(f"Native Nsight report: {report_path}")
        print(f"Raw Nsight CSV: {output_path}")
        return str(output_path)

    except FileNotFoundError as exc:
        raise RuntimeError(
            "ncu was not found in PATH. Install NVIDIA Nsight Compute and ensure "
            "the CLI is available in the profiling environment."
        ) from exc

if __name__ == '__main__':
    # Demo: generate synthetic Nsight report and visualize
    print("Generating Nsight analysis with demo data...")
    report = NsightReport._demo_report()
    visualize_nsight_report(report, 'nsight_analysis_demo.png')
    report.to_json('nsight_report.json')
