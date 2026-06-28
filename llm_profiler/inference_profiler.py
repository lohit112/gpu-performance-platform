"""
llm_profiler/inference_profiler.py
─────────────────────────────────────────────────────────────────────────────
First-principles LLM inference profiler.

Given: model size, context length, batch size, precision, GPU
Outputs: VRAM breakdown, KV cache, prefill/decode latency, tokens/sec,
         cost per million tokens, bottleneck classification.

WHY FIRST PRINCIPLES MATTERS:
  Tools like LLM-Perf and Hugging Face benchmarks give you numbers.
  This module gives you the MATH behind those numbers — and that's what
  NVIDIA engineers ask about in interviews.

  If you can derive "GPT-3 (175B) decode at batch=1 is purely memory-bound
  because arithmetic intensity < 1 FLOP/Byte on any GPU" from scratch,
  you understand LLM inference at an architectural level.

KEY FORMULAS:
  Model weights VRAM:
    bytes = num_params × bytes_per_param
    e.g. 7B FP16 = 7e9 × 2 = 14 GB

  KV Cache VRAM per token per layer:
    bytes = 2 × num_heads × head_dim × bytes_per_elem
    (factor 2 = K and V)
    Total KV cache = 2 × num_layers × num_heads × head_dim × seq_len × batch × bytes

  Prefill FLOPs (attention dominant):
    FLOPs ≈ 2 × num_layers × seq_len² × hidden_dim   (attention self-product)
           + 2 × num_layers × seq_len × 4 × hidden_dim²  (FFN)
    This is O(N²) in sequence length — confirms quadratic scaling.

  Decode FLOPs (per token generated):
    FLOPs ≈ 2 × num_layers × seq_len × hidden_dim   (attention over KV cache)
           + 2 × num_layers × 4 × hidden_dim²         (FFN)
    Attention term is O(seq_len) — grows with generation length.

  Decode Memory Bottleneck:
    Bytes moved per decode step = model_weights + KV_cache
    Arithmetic Intensity ≈ 2 × hidden_dim × 4 / 16 bytes = ~hidden_dim/4
    For GPT-2 (hidden=768): OI ≈ 192  → compute-bound on T4
    For Llama-7B (hidden=4096): OI ≈ 1024 → compute-bound
    BUT at batch=1: effective OI = OI / batch → memory-bound for small batches
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# ─── GPU Hardware Database ─────────────────────────────────────────────────────
@dataclass
class GPUSpec:
    name:              str
    vram_gb:           float
    bandwidth_gbps:    float
    fp16_tflops:       float
    fp32_tflops:       float
    fp8_tflops:        float
    tensor_fp16_tflops: float
    tdp_watts:         int
    cost_per_hour_usd: float   # cloud on-demand (approximate, 2025)
    architecture:      str

GPU_CATALOG = {
    'T4':      GPUSpec('T4',      16,   320,   65,    8.1,  130,   65,    70,   0.53,  'Turing'),
    'L4':      GPUSpec('L4',      24,   300,   121,   30.3, 242,   121,   72,   0.80,  'Ada'),
    'A100_40': GPUSpec('A100-40', 40,  1555,   312,   19.5, 624,   312,   400,  3.21,  'Ampere'),
    'A100_80': GPUSpec('A100-80', 80,  2000,   312,   19.5, 624,   312,   400,  3.21,  'Ampere'),
    'H100_SXM':GPUSpec('H100 SXM',80,  3350,  1979,   67.0,3958,  1979,   700,  5.12,  'Hopper'),
    'H100_PCIe':GPUSpec('H100 PCIe',80, 2000, 1513,   48.0,3026,  1513,   350,  3.80,  'Hopper'),
    'RTX4090': GPUSpec('RTX 4090', 24,  1008,  330,   82.6, 661,   330,   450,  2.50,  'Ada'),
    'RTX5090': GPUSpec('RTX 5090', 32,  1792,  838,  109.0,1676,   838,   575,  3.50,  'Blackwell'),
    'B200':    GPUSpec('B200 SXM', 192, 8000,  9000, 180.0,18000, 9000,  1000,  8.00,  'Blackwell'),
}


# ─── LLM Model Configs ────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name:           str
    num_params:     float      # total parameters (billions → stored as float)
    num_layers:     int
    hidden_dim:     int
    num_heads:      int
    head_dim:       int
    ffn_multiplier: float = 4.0    # FFN inner dim = ffn_multiplier × hidden_dim
    num_kv_heads:   int = 0        # for GQA (0 = same as num_heads = MHA)
    vocab_size:     int = 32000

    def __post_init__(self):
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads

MODEL_CATALOG = {
    'gpt2':       ModelConfig('GPT-2',         0.124, 12,   768, 12,  64),
    'gpt2_xl':    ModelConfig('GPT-2 XL',      1.5,   48,  1600, 25,  64),
    'llama_7b':   ModelConfig('Llama-2 7B',    7.0,   32,  4096, 32, 128, num_kv_heads=32),
    'llama_13b':  ModelConfig('Llama-2 13B',   13.0,  40,  5120, 40, 128, num_kv_heads=40),
    'llama_70b':  ModelConfig('Llama-2 70B',   70.0,  80,  8192, 64, 128, num_kv_heads=8),
    'llama3_8b':  ModelConfig('Llama-3 8B',    8.0,   32,  4096, 32, 128, num_kv_heads=8),
    'llama3_70b': ModelConfig('Llama-3 70B',   70.0,  80,  8192, 64, 128, num_kv_heads=8),
    'mistral_7b': ModelConfig('Mistral 7B',    7.0,   32,  4096, 32, 128, num_kv_heads=8),
    'mixtral_8x7b':ModelConfig('Mixtral 8×7B', 46.7,  32,  4096, 32, 128, num_kv_heads=8),
    'gpt4_est':   ModelConfig('GPT-4 (est.)',  220.0, 96, 12288,128, 128, num_kv_heads=128),
}

PRECISION_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'fp8': 1, 'int4': 0.5}


# ─── Profiler Output ──────────────────────────────────────────────────────────
@dataclass
class InferenceProfile:
    """Complete inference performance profile."""
    model:          ModelConfig
    gpu:            GPUSpec
    context_len:    int
    batch_size:     int
    precision:      str

    # ── VRAM breakdown (GB) ─────────────────────────────────────────────────
    weights_gb:         float = 0.0
    kv_cache_gb:        float = 0.0
    activations_gb:     float = 0.0
    framework_overhead_gb: float = 1.5    # PyTorch/TRT overhead
    total_vram_gb:      float = 0.0
    fits_on_gpu:        bool  = False
    num_gpus_required:  int   = 1

    # ── FLOPs analysis ──────────────────────────────────────────────────────
    prefill_flops:      float = 0.0    # total FLOPs for prefill
    decode_flops_per_tok: float = 0.0  # FLOPs per generated token

    # ── Arithmetic intensity ─────────────────────────────────────────────────
    prefill_oi:         float = 0.0    # FLOP/Byte (prefill)
    decode_oi:          float = 0.0    # FLOP/Byte (decode, batch-normalized)

    # ── Latency estimates ────────────────────────────────────────────────────
    prefill_latency_ms: float = 0.0
    decode_latency_ms_per_tok: float = 0.0

    # ── Throughput ──────────────────────────────────────────────────────────
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec:  float = 0.0  # per GPU
    decode_tokens_per_sec_batch: float = 0.0  # across batch

    # ── Bottleneck ──────────────────────────────────────────────────────────
    prefill_bound:  str = ''    # 'COMPUTE' | 'MEMORY'
    decode_bound:   str = ''
    ridge_point:    float = 0.0

    # ── Cost ────────────────────────────────────────────────────────────────
    cost_per_1M_input_tokens:  float = 0.0
    cost_per_1M_output_tokens: float = 0.0
    gpu_utilization_pct:       float = 0.0

    # ── Recommendations ─────────────────────────────────────────────────────
    recommendations: list[str] = field(default_factory=list)


class LLMInferenceProfiler:
    """
    First-principles LLM inference profiler.
    All math is explicit with comments — defensible in interviews.
    """

    def __init__(self, mfu_estimate: float = 0.35):
        """
        mfu_estimate: model FLOP utilization fraction (realistic: 0.3–0.5).
        cuBLAS-level GEMM gets 0.8+ but full transformer has overhead.
        0.35 is empirically accurate for well-optimized PyTorch + Flash Attention.
        """
        self.mfu = mfu_estimate

    def profile(
        self,
        model_key:   str,
        gpu_key:     str,
        context_len: int,
        batch_size:  int,
        precision:   str = 'fp16',
        output_len:  int = 256,
    ) -> InferenceProfile:

        model = MODEL_CATALOG[model_key]
        gpu   = GPU_CATALOG[gpu_key]
        p     = InferenceProfile(model, gpu, context_len, batch_size, precision)
        bpp   = PRECISION_BYTES[precision]  # bytes per parameter

        # ── 1. VRAM BREAKDOWN ────────────────────────────────────────────────
        # Weights
        p.weights_gb = model.num_params * 1e9 * bpp / 1e9

        # KV Cache:
        #   Per layer, per token: K and V each have shape [num_kv_heads, head_dim]
        #   = 2 × num_kv_heads × head_dim × bytes_per_elem
        #   Total: multiply by num_layers, seq_len, batch_size
        kv_bytes_per_token_per_layer = (
            2 *                         # K and V
            model.num_kv_heads *
            model.head_dim *
            bpp
        )
        p.kv_cache_gb = (
            kv_bytes_per_token_per_layer *
            model.num_layers *
            context_len *
            batch_size
        ) / 1e9

        # Activations (rough): one forward pass holds ~seq_len × hidden × 2 (checkpointing)
        p.activations_gb = (
            batch_size * context_len * model.hidden_dim * bpp * 2
        ) / 1e9

        p.total_vram_gb = (
            p.weights_gb +
            p.kv_cache_gb +
            p.activations_gb +
            p.framework_overhead_gb
        )

        p.fits_on_gpu = p.total_vram_gb <= gpu.vram_gb
        p.num_gpus_required = max(1, math.ceil(p.total_vram_gb / gpu.vram_gb))

        # ── 2. FLOPs ANALYSIS ────────────────────────────────────────────────
        L  = model.num_layers
        S  = context_len
        H  = model.hidden_dim
        Hd = model.head_dim
        Nh = model.num_heads
        F  = int(model.ffn_multiplier * H)   # FFN inner dim

        # Prefill FLOPs (total for all S tokens in parallel)
        # Self-attention QKV projections: 3 × (S × H) × H = 6SH² per layer
        attn_proj_flops = 6 * S * H * H
        # Attention scores: S × S × Nh × Hd = S² × H (since Nh × Hd = H)
        attn_score_flops = 2 * S * S * H
        # FFN: 2 layers of (S × H × F): 2 × 2 × S × H × F
        ffn_flops = 4 * S * H * F
        prefill_flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
        p.prefill_flops = L * prefill_flops_per_layer * batch_size

        # Decode FLOPs (per token, attending over full KV cache of length S)
        # Single new token, attending over S cached tokens
        dec_attn_proj  = 6 * H * H         # QKV for 1 new token
        dec_attn_score = 2 * S * H         # attending over S cached positions
        dec_ffn        = 4 * H * F
        p.decode_flops_per_tok = L * (dec_attn_proj + dec_attn_score + dec_ffn) * batch_size

        # ── 3. ARITHMETIC INTENSITY ──────────────────────────────────────────
        # Prefill: memory traffic = weights loaded once + KV written
        prefill_bytes = p.weights_gb * 1e9 + p.kv_cache_gb * 1e9 / 2
        p.prefill_oi  = p.prefill_flops / max(prefill_bytes, 1)

        # Decode: each step loads ALL weights + KV cache
        decode_bytes_per_step = (p.weights_gb + p.kv_cache_gb) * 1e9
        p.decode_oi = p.decode_flops_per_tok / max(decode_bytes_per_step, 1)

        # Ridge point
        # gpu.*_tflops fields are in TFLOPS; convert to GFLOPS (*1000) for consistency
        peak_gflops = (
            gpu.tensor_fp16_tflops * 1000 if precision in ('fp16', 'bf16')
            else gpu.fp32_tflops * 1000
        )
        # Ridge point: FLOP/Byte where compute roof meets memory roof
        # = peak_FLOPS / peak_BW = (peak_gflops*1e9) / (bandwidth_gbps*1e9)
        p.ridge_point = peak_gflops / gpu.bandwidth_gbps   # GFLOPS / (GB/s) = FLOP/Byte
        p.prefill_bound = 'COMPUTE' if p.prefill_oi > p.ridge_point else 'MEMORY'
        p.decode_bound  = 'COMPUTE' if p.decode_oi  > p.ridge_point else 'MEMORY'

        # ── 4. LATENCY + THROUGHPUT ─────────────────────────────────────────
        eff_peak_gflops = peak_gflops * self.mfu

        # Prefill: compute-bound determines latency when OI high, BW-bound otherwise
        if p.prefill_bound == 'COMPUTE':
            prefill_sec = p.prefill_flops / (eff_peak_gflops * 1e9)
        else:
            prefill_sec = prefill_bytes / (gpu.bandwidth_gbps * 1e9)
        p.prefill_latency_ms = prefill_sec * 1000

        # Decode: nearly always memory-bound at small batch
        if p.decode_bound == 'COMPUTE':
            decode_sec = p.decode_flops_per_tok / (eff_peak_gflops * 1e9)
        else:
            decode_sec = decode_bytes_per_step / (gpu.bandwidth_gbps * 1e9)
        p.decode_latency_ms_per_tok = decode_sec * 1000

        p.prefill_tokens_per_sec     = (batch_size * S) / max(prefill_sec, 1e-9)
        p.decode_tokens_per_sec      = 1.0 / max(decode_sec, 1e-9)
        p.decode_tokens_per_sec_batch = p.decode_tokens_per_sec * batch_size

        # ── 5. COST ANALYSIS ─────────────────────────────────────────────────
        # Effective GPUs needed
        n_gpus = p.num_gpus_required
        cost_per_sec = (gpu.cost_per_hour_usd * n_gpus) / 3600.0

        # Cost per 1M input tokens (prefill)
        if p.prefill_tokens_per_sec > 0:
            secs_per_1M = 1e6 / p.prefill_tokens_per_sec
            p.cost_per_1M_input_tokens = secs_per_1M * cost_per_sec
        else:
            p.cost_per_1M_input_tokens = 999.0

        # Cost per 1M output tokens (decode)
        if p.decode_tokens_per_sec_batch > 0:
            secs_per_1M_out = 1e6 / p.decode_tokens_per_sec_batch
            p.cost_per_1M_output_tokens = secs_per_1M_out * cost_per_sec
        else:
            p.cost_per_1M_output_tokens = 999.0

        # GPU utilization (compute)
        p.gpu_utilization_pct = min(100.0, self.mfu * 100)

        # ── 6. RECOMMENDATIONS ───────────────────────────────────────────────
        p.recommendations = self._generate_recommendations(p)

        return p

    def _generate_recommendations(self, p: InferenceProfile) -> list[str]:
        recs = []
        model, gpu = p.model, p.gpu

        if not p.fits_on_gpu:
            recs.append(
                f"[ERROR] MODEL DOES NOT FIT: {p.total_vram_gb:.1f} GB required, "
                f"{gpu.vram_gb} GB available. Need {p.num_gpus_required} GPUs. "
                f"Solutions: tensor parallelism (split weight matrices across GPUs), "
                f"pipeline parallelism (split layers), or quantize to int4/fp8."
            )
        else:
            headroom = gpu.vram_gb - p.total_vram_gb
            recs.append(
                f"Model fits with {headroom:.1f} GB headroom. "
                f"({p.weights_gb:.1f}GB weights + {p.kv_cache_gb:.1f}GB KV + "
                f"{p.activations_gb:.1f}GB activations)"
            )

        if p.decode_bound == 'MEMORY':
            recs.append(
                f"Decode is memory-bound (OI={p.decode_oi:.2f} < ridge={p.ridge_point:.0f}). "
                f"Classic small-batch LLM symptom. GPU bandwidth ({gpu.bandwidth_gbps} GB/s) "
                f"is the bottleneck, NOT compute. "
                f"Fix: increase batch size (saturates BW), use continuous batching, "
                f"or quantize weights to reduce bytes loaded per step."
            )

        if p.batch_size == 1:
            recs.append(
                f"Batch size = 1: GPU utilization is very low. "
                f"In decode, each step loads {p.weights_gb:.1f}GB of weights "
                f"to compute just {p.decode_flops_per_tok/1e9:.1f}B FLOPs. "
                f"Try batch_size=16: same weights loaded, 16× the throughput."
            )

        if p.kv_cache_gb > p.weights_gb:
            recs.append(
                f"KV cache ({p.kv_cache_gb:.1f}GB) EXCEEDS WEIGHTS ({p.weights_gb:.1f}GB). "
                f"Context length {p.context_len} with batch {p.batch_size} is memory-heavy. "
                f"Consider: GQA (Llama-3 uses 8 KV heads instead of 32), "
                f"sliding window attention (Mistral), or PagedAttention (vLLM)."
            )

        if p.prefill_latency_ms > 1000:
            recs.append(
                f"Long prefill: {p.prefill_latency_ms:.0f}ms for {p.context_len}-token context. "
                f"This is O(N²) in sequence length. At 2× context → 4× prefill time. "
                f"For real-time applications, chunk prefill into smaller pieces (speculative prefill)."
            )

        if p.decode_tokens_per_sec_batch < 10:
            recs.append(
                f"Low decode throughput: {p.decode_tokens_per_sec_batch:.1f} tok/s. "
                f"Below interactive speed (30+ tok/s for text streaming). "
                f"Consider: int8 quantization (2× weights reduced → 2× throughput), "
                f"speculative decoding (draft model), or continuous batching."
            )

        recs.append(
            f"Cost: ${p.cost_per_1M_input_tokens:.2f}/1M input tokens, "
            f"${p.cost_per_1M_output_tokens:.2f}/1M output tokens "
            f"on {gpu.name} @ ${gpu.cost_per_hour_usd}/hr. "
            f"({'×' + str(p.num_gpus_required) + ' GPUs' if p.num_gpus_required > 1 else '1 GPU'})"
        )

        return recs

    def batch_profile(
        self,
        model_key: str,
        gpu_key:   str,
        context_len: int = 2048,
        precision: str = 'fp16',
    ) -> list[InferenceProfile]:
        """Profile across batch sizes 1→64 to find optimal operating point."""
        results = []
        for bs in [1, 2, 4, 8, 16, 32, 64]:
            p = self.profile(model_key, gpu_key, context_len, bs, precision)
            results.append(p)
        return results


def visualize_inference_profile(
    profiles: list[InferenceProfile],
    output_path: str = 'llm_inference_profile.png',
) -> None:
    """
    4-panel LLM inference visualization:
    A: VRAM breakdown by component
    B: Decode throughput vs batch size
    C: Latency breakdown (prefill + decode)
    D: Cost per million tokens vs batch size
    """
    if not profiles:
        return

    BG, SURFACE, BORDER = '#0a0a0f', '#111118', '#1e1e2e'
    ref = profiles[0]

    fig = plt.figure(figsize=(22, 12), facecolor=BG)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)

    def style_ax(ax):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors='#8888aa', labelsize=8)
        for lab in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            lab.set_color('#e8e8f0' if lab == ax.title else '#8888aa')
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(axis='y', color='#1a1a28', linewidth=0.5)
        ax.set_axisbelow(True)

    # ── Panel A: VRAM breakdown ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    categories  = ['Weights', 'KV Cache', 'Activations', 'Framework\nOverhead']
    values      = [ref.weights_gb, ref.kv_cache_gb, ref.activations_gb, ref.framework_overhead_gb]
    colors      = ['#3d9abf', '#f5c842', '#e8503a', '#4a4a6a']
    bars = ax.bar(categories, values, color=colors, alpha=0.85, zorder=3)
    ax.axhline(ref.gpu.vram_gb, color='#76b900', linestyle='--', linewidth=1.5, label=f'GPU VRAM ({ref.gpu.vram_gb}GB)', zorder=4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f'{val:.1f}GB', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    ax.set_ylabel('VRAM (GB)', fontsize=9)
    ax.set_title(f'A  VRAM Breakdown\n{ref.model.name} | {ref.precision.upper()} | ctx={ref.context_len} | bs={ref.batch_size}',
                 fontsize=9, fontweight='bold', pad=6)
    style_ax(ax)

    # ── Panel B: Decode throughput vs batch size ─────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    batch_sizes   = [p.batch_size for p in profiles]
    decode_tps    = [p.decode_tokens_per_sec_batch for p in profiles]
    bound_colors  = ['#e8503a' if p.decode_bound == 'MEMORY' else '#3abf7a' for p in profiles]
    scatter_pts   = ax.scatter(batch_sizes, decode_tps, c=bound_colors, s=80, zorder=5, edgecolors='white', linewidths=0.6)
    ax.plot(batch_sizes, decode_tps, color='#3d9abf', linewidth=1.5, alpha=0.6, zorder=4)
    ax.axhline(30, color='#f5c842', linestyle=':', linewidth=1.2, label='30 tok/s (interactive)', zorder=3)
    for bs, tps in zip(batch_sizes, decode_tps):
        ax.annotate(f'{tps:.0f}', (bs, tps), xytext=(0, 8), textcoords='offset points',
                    ha='center', color='#aaaacc', fontsize=7)
    ax.set_xscale('log', base=2); ax.set_xticks(batch_sizes); ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    red_patch   = mpatches.Patch(color='#e8503a', label='Memory-bound')
    green_patch = mpatches.Patch(color='#3abf7a', label='Compute-bound')
    ax.legend(handles=[red_patch, green_patch, ax.lines[1]], fontsize=8,
              framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    ax.set_xlabel('Batch Size', fontsize=9)
    ax.set_ylabel('Decode Tokens/sec (batch total)', fontsize=9)
    ax.set_title(f'B  Decode Throughput vs Batch Size\n{ref.gpu.name} | {ref.model.name}',
                 fontsize=9, fontweight='bold', pad=6)
    style_ax(ax)

    # ── Panel C: Latency breakdown ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    prefill_lats = [p.prefill_latency_ms for p in profiles]
    decode_lats  = [p.decode_latency_ms_per_tok * 256 for p in profiles]  # 256 output tokens
    x = np.arange(len(profiles))
    w = 0.35
    ax.bar(x - w/2, prefill_lats, w, color='#3d9abf', alpha=0.85, label='Prefill latency (ms)')
    ax.bar(x + w/2, decode_lats,  w, color='#f5c842', alpha=0.85, label='Decode 256 tok (ms)')
    ax.set_xticks(x); ax.set_xticklabels([f'bs={p.batch_size}' for p in profiles], fontsize=8)
    ax.set_ylabel('Latency (ms)', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    ax.set_title('C  Latency Breakdown\n(Prefill + Decode 256 tokens)',
                 fontsize=9, fontweight='bold', pad=6)
    style_ax(ax)

    # ── Panel D: Cost per million tokens ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    input_costs  = [p.cost_per_1M_input_tokens  for p in profiles]
    output_costs = [p.cost_per_1M_output_tokens for p in profiles]
    ax.plot(batch_sizes, input_costs,  'o-', color='#3d9abf', linewidth=2, markersize=6, label='Input (prefill) $/1M tok')
    ax.plot(batch_sizes, output_costs, 's--', color='#f5c842', linewidth=2, markersize=6, label='Output (decode) $/1M tok')
    ax.set_xscale('log', base=2); ax.set_xticks(batch_sizes); ax.set_xticklabels(batch_sizes)
    ax.set_xlabel('Batch Size', fontsize=9)
    ax.set_ylabel('Cost per 1M Tokens (USD)', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=SURFACE, edgecolor=BORDER, labelcolor='#aaaacc')
    ax.set_title(f'D  Cost Analysis — {ref.gpu.name} @ ${ref.gpu.cost_per_hour_usd}/hr\n'
                 f'(×{ref.num_gpus_required} GPUs needed for this config)',
                 fontsize=9, fontweight='bold', pad=6)
    style_ax(ax)

    model_name = ref.model.name
    gpu_name   = ref.gpu.name
    fig.suptitle(
        f'LLM Inference Profile — {model_name} on {gpu_name} | {ref.precision.upper()} | ctx={ref.context_len}',
        color='#e8e8f0', fontsize=14, fontweight='bold', y=0.99,
    )

    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"LLM inference profile saved → {output_path}")


def print_profile_report(p: InferenceProfile) -> None:
    """Print a human-readable profile report."""
    fits_str = "fits" if p.fits_on_gpu else f"need {p.num_gpus_required} GPUs"
    print()
    print(f"LLM Inference Profile")
    print(f"  Model:     {p.model.name}  |  GPU: {p.gpu.name}")
    print(f"  Precision: {p.precision}  |  Context: {p.context_len}  |  Batch: {p.batch_size}")
    print()
    print("VRAM Breakdown")
    print(f"  Weights:       {p.weights_gb:>6.1f} GB")
    print(f"  KV Cache:      {p.kv_cache_gb:>6.1f} GB")
    print(f"  Activations:   {p.activations_gb:>6.1f} GB")
    print(f"  Overhead:      {p.framework_overhead_gb:>6.1f} GB")
    print(f"  Total:         {p.total_vram_gb:>6.1f} GB  ({fits_str})")
    print()
    print("Performance")
    print(f"  Prefill:     {p.prefill_latency_ms:>7.1f} ms   ({p.prefill_bound}-bound, OI={p.prefill_oi:.1f} FLOP/Byte)")
    print(f"  Decode:      {p.decode_latency_ms_per_tok:>7.1f} ms/tok ({p.decode_bound}-bound, OI={p.decode_oi:.2f} FLOP/Byte)")
    print(f"  Throughput:  {p.decode_tokens_per_sec_batch:>6.0f} tok/s (batch total)")
    print(f"  Ridge point: {p.ridge_point:.0f} FLOP/Byte")
    print()
    print("Cost")
    print(f"  Input:  ${p.cost_per_1M_input_tokens:>7.2f} / 1M tokens")
    print(f"  Output: ${p.cost_per_1M_output_tokens:>7.2f} / 1M tokens")
    print()
    print("Recommendations")
    for rec in p.recommendations:
        # Strip leading emoji if present
        clean = rec.strip()
        words = clean.split()
        line, lines = '', []
        for w in words:
            if len(line) + len(w) > 70:
                lines.append(line)
                line = w
            else:
                line += (' ' if line else '') + w
        if line: lines.append(line)
        for i, l in enumerate(lines):
            prefix = "  - " if i == 0 else "    "
            print(f"{prefix}{l}")
        print()


if __name__ == '__main__':
    profiler = LLMInferenceProfiler(mfu_estimate=0.35)

    print("\n=== Single Profile: Llama-2 7B on T4 ===")
    p = profiler.profile('llama_7b', 'T4', context_len=2048, batch_size=1, precision='fp16')
    print_profile_report(p)

    print("\n=== Batch sweep: Llama-2 7B on A100-80 ===")
    profiles = profiler.batch_profile('llama_7b', 'A100_80', context_len=4096, precision='fp16')
    visualize_inference_profile(profiles, 'llm_inference_profile.png')

    print("\n=== Large model: Llama-2 70B on H100 ===")
    p70b = profiler.profile('llama_70b', 'H100_SXM', context_len=4096, batch_size=4, precision='fp16')
    print_profile_report(p70b)
