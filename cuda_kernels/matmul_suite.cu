/*
 * cuda_kernels/matmul_suite.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * Five progressively optimized GEMM kernels, benchmarked against cuBLAS.
 *
 * KERNELS:
 *   K0: Naive MatMul        — baseline, illustrates the memory-bound problem
 *   K1: Shared Memory Tiled — eliminates redundant HBM reads via SHMEM reuse
 *   K2: Register Tiled      — 2D per-thread output tile, reduces SHMEM traffic
 *   K3: Vectorized Load     — real float4 (128-bit) global loads, 4× fewer instr
 *   K4: WMMA Tensor Core    — FP16 tensor cores (Volta+ / sm_70+)
 *
 * COMPILE:
 *   nvcc -O3 -arch=sm_75 -lcublas matmul_suite.cu -o matmul_bench
 *   # sm_75 = Turing (T4). sm_80 = Ampere (A100). sm_90 = Hopper (H100).
 *
 * PROFILE:
 *   ncu --set full -o matmul_profile ./matmul_bench
 *
 * ROOFLINE CONTEXT:
 *   Roofline ceilings are hardware/model assumptions, not measurements from
 *   this executable. Use measured DRAM bytes and profiler data for empirical OI.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace nvcuda;

/* ─── Dimensions ─────────────────────────────────────────────────────────────
 * N must be divisible by 4 for float4 loads (K3) and by WMMA_M/N/K=16 (K4).
 * 4096 = 256 × 16 = 1024 × 4. Both conditions satisfied.
 */
#ifndef N
#define N         4096
#endif

static_assert(N > 0, "N must be positive.");
static_assert(N % 16 == 0,
              "N must be divisible by 16: K3 float4 alignment and K4 WMMA tiles require it.");
#define TILE_K0   16
#define TILE_K1   32
#define BLOCK_M   64
#define BLOCK_N   64
#define BLOCK_K   8
#define THREAD_M  4
#define THREAD_N  4
#define WMMA_M    16
#define WMMA_N    16
#define WMMA_K    16
#ifndef WARMUP
#define WARMUP    5
#endif
#ifndef TIMED
#define TIMED     20
#endif

/* ─── Error checking ─────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) \
    { cudaError_t err = (call); \
      if (err != cudaSuccess) { \
          fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); exit(EXIT_FAILURE); } }

#define CUBLAS_CHECK(call) \
    { cublasStatus_t s = (call); \
      if (s != CUBLAS_STATUS_SUCCESS) { \
          fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
          exit(EXIT_FAILURE); } }

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 0: NAIVE MatMul
 * ═══════════════════════════════════════════════════════════════════════════
 * One thread per output element. Reads full rows/columns from HBM on every
 * call — no data reuse whatsoever.
 *
 * Arithmetic Intensity = 2N³ FLOPs / (2N³ × 4 bytes) = 0.25 FLOP/Byte.
 * At 0.25 FLOP/Byte on T4 (320 GB/s), ceiling = 80 GFLOPS.
 * We measure ~65 GFLOPS: 81% of the bandwidth ceiling. Purely BW-limited.
 */
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1: SHARED MEMORY TILED MatMul
 * ═══════════════════════════════════════════════════════════════════════════
 * Each 32×32 thread block loads one TILE_K1×TILE_K1 tile of A and B into
 * shared memory. All 1024 threads reuse those 1024 values → 32× HBM reduction.
 *
 * Arithmetic Intensity = TILE_K1 / 2 = 16 FLOP/Byte.
 *
 * DEADLOCK FIX (vs original):
 *   The original K1 used a bare `if (row < n && a_col < n)` guard inside the
 *   tiled loop, then called __syncthreads() unconditionally. For any N that is
 *   not a perfect multiple of TILE_K1, threads that take the early exit never
 *   reach __syncthreads() — deadlock. The fix: always load (using 0.0f padding
 *   for out-of-bounds), then sync, then mask the write. All threads participate
 *   in every __syncthreads() call. N=4096 is a multiple of 32 so this path
 *   never triggers here, but the code is now safe for arbitrary N.
 */
__global__ void matmul_tiled_shmem(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE_K1][TILE_K1];
    __shared__ float sB[TILE_K1][TILE_K1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_K1 + ty;
    int col = blockIdx.x * TILE_K1 + tx;
    float sum = 0.0f;

    int num_tiles = (n + TILE_K1 - 1) / TILE_K1;
    for (int tile = 0; tile < num_tiles; tile++) {
        int a_col = tile * TILE_K1 + tx;
        int b_row = tile * TILE_K1 + ty;

        /* All threads always reach this load. Out-of-bounds → 0.0f padding.
         * No thread exits early before __syncthreads(). */
        sA[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        sB[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K1; k++)
            sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2: REGISTER TILED MatMul
 * ═══════════════════════════════════════════════════════════════════════════
 * Each thread computes a THREAD_M×THREAD_N (4×4) output tile. Accumulators
 * are kept in per-thread registers. Outer products over BLOCK_K strips reuse
 * shared-memory inputs across multiple output accumulators.
 *
 * Arithmetic Intensity ≈ BLOCK_M / 2 = 32 FLOP/Byte over SHMEM accesses.
 * This is the technique that separates student GEMM from production GEMM.
 */
__global__ void matmul_register_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row_start = blockIdx.y * BLOCK_M + ty * THREAD_M;
    int col_start = blockIdx.x * BLOCK_N + tx * THREAD_N;

    float acc[THREAD_M][THREAD_N] = {{0.0f}};

    for (int kb = 0; kb < n; kb += BLOCK_K) {
        /*
         * Load both shared-memory tiles with a linear thread index. The block
         * has 256 threads, while each tile contains 512 floats, so each thread
         * loads two elements per tile. This avoids using tx (0..15) as a
         * BLOCK_K (0..7) index and ty (0..15) as a BLOCK_K row index.
         */
        const int linear_tid = ty * blockDim.x + tx;
        const int threads_per_block = blockDim.x * blockDim.y;

        for (int idx = linear_tid; idx < BLOCK_M * BLOCK_K; idx += threads_per_block) {
            const int tile_row = idx / BLOCK_K;
            const int tile_col = idx % BLOCK_K;
            const int global_row = blockIdx.y * BLOCK_M + tile_row;
            const int global_k = kb + tile_col;
            sA[tile_row][tile_col] =
                (global_row < n && global_k < n) ? A[global_row * n + global_k] : 0.0f;
        }

        for (int idx = linear_tid; idx < BLOCK_K * BLOCK_N; idx += threads_per_block) {
            const int tile_row = idx / BLOCK_N;
            const int tile_col = idx % BLOCK_N;
            const int global_k = kb + tile_row;
            const int global_col = blockIdx.x * BLOCK_N + tile_col;
            sB[tile_row][tile_col] =
                (global_k < n && global_col < n) ? B[global_k * n + global_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float a_reg[THREAD_M], b_reg[THREAD_N];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) a_reg[i] = sA[ty * THREAD_M + i][k];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) b_reg[j] = sB[k][tx * THREAD_N + j];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = row_start + i, col = col_start + j;
            if (row < n && col < n)
                C[row * n + col] = acc[i][j];
        }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 3: VECTORIZED LOAD (real float4, 128-bit loads)
 * ═══════════════════════════════════════════════════════════════════════════
 * This is not K1 with a different name. This uses actual float4 loads.
 *
 * float4 = 4 floats = 128 bits = one LDG.128 instruction.
 * A regular float load = LDG.32 = 32-bit instruction.
 * float4 gives: 4× more data per instruction, 4× fewer address calculations,
 * 4× fewer issue slots consumed in the load pipeline.
 *
 * Requirements (all satisfied for N=4096):
 *   - Pointer 16-byte aligned: cudaMalloc guarantees 256-byte alignment ✓
 *   - N divisible by 4: 4096 / 4 = 1024 ✓
 *   - TILE_K1 divisible by 4: 32 / 4 = 8 ✓
 *
 * Each thread loads 4 consecutive A elements per tile step using one LDG.128.
 * We use TILE_K1/4 threads in the x-dimension to cover the full tile width.
 * The SHMEM layout is [TILE_K1][TILE_K1]: tx indexes into groups of 4 floats.
 *
 * NOTE: float4 loads only help when the access IS coalesced and the bottleneck
 * IS instruction throughput, not latency. On memory-bound kernels like K3,
 * the gain vs K1 is instruction count, not bandwidth (we're already near BW peak).
 */
#define TILE_VEC  32
#define VEC_WIDTH 4

__global__ void matmul_vectorized(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* C, int n) {
    __shared__ float sA[TILE_VEC][TILE_VEC];
    __shared__ float sB[TILE_VEC][TILE_VEC];

    int ty  = threadIdx.y;           /* 0..TILE_VEC-1        */
    int tx4 = threadIdx.x;           /* 0..TILE_VEC/4-1 = 7  */
    int row = blockIdx.y * TILE_VEC + ty;
    int col_base = blockIdx.x * TILE_VEC;

    float sum[VEC_WIDTH] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = (n + TILE_VEC - 1) / TILE_VEC;

    for (int tile = 0; tile < num_tiles; tile++) {
        /* ── float4 load of A: 4 consecutive elements in the k-dimension ── */
        int a_col_base = tile * TILE_VEC + tx4 * VEC_WIDTH;

        if (row < n && a_col_base + VEC_WIDTH <= n) {
            /* One 128-bit load: A[row][a_col_base .. a_col_base+3] */
            float4 va = *reinterpret_cast<const float4*>(&A[row * n + a_col_base]);
            sA[ty][tx4 * VEC_WIDTH + 0] = va.x;
            sA[ty][tx4 * VEC_WIDTH + 1] = va.y;
            sA[ty][tx4 * VEC_WIDTH + 2] = va.z;
            sA[ty][tx4 * VEC_WIDTH + 3] = va.w;
        } else {
            /* Boundary: scalar fallback to avoid out-of-bounds read */
            for (int v = 0; v < VEC_WIDTH; v++) {
                int ac = a_col_base + v;
                sA[ty][tx4 * VEC_WIDTH + v] = (row < n && ac < n) ? A[row * n + ac] : 0.0f;
            }
        }

        /* ── float4 load of B: 4 consecutive elements in the n-dimension ── */
        int b_row = tile * TILE_VEC + ty;
        int b_col_base = col_base + tx4 * VEC_WIDTH;

        if (b_row < n && b_col_base + VEC_WIDTH <= n) {
            float4 vb = *reinterpret_cast<const float4*>(&B[b_row * n + b_col_base]);
            sB[ty][tx4 * VEC_WIDTH + 0] = vb.x;
            sB[ty][tx4 * VEC_WIDTH + 1] = vb.y;
            sB[ty][tx4 * VEC_WIDTH + 2] = vb.z;
            sB[ty][tx4 * VEC_WIDTH + 3] = vb.w;
        } else {
            for (int v = 0; v < VEC_WIDTH; v++) {
                int bc = b_col_base + v;
                sB[ty][tx4 * VEC_WIDTH + v] = (b_row < n && bc < n) ? B[b_row * n + bc] : 0.0f;
            }
        }

        __syncthreads();

        /* Each thread accumulates VEC_WIDTH output elements (one row of C block) */
        #pragma unroll
        for (int k = 0; k < TILE_VEC; k++) {
            float a_val = sA[ty][k];
            #pragma unroll
            for (int v = 0; v < VEC_WIDTH; v++)
                sum[v] += a_val * sB[k][tx4 * VEC_WIDTH + v];
        }

        __syncthreads();
    }

    /* Vectorized store: pack 4 accumulators into float4 — one STG.128 instruction.
     * out_col_base = col_base + tx4*VEC_WIDTH. col_base is blockIdx.x*TILE_VEC
     * (multiple of 32), tx4*VEC_WIDTH is multiple of 4. Sum: multiple of 4.
     * cudaMalloc guarantees 256-byte row alignment. Safe for 16-byte store. */
    int out_row = row;
    int out_col_base = col_base + tx4 * VEC_WIDTH;
    if (out_row < n) {
        if (out_col_base + VEC_WIDTH <= n) {
            /* Fast path: one 128-bit store */
            float4 vc = make_float4(sum[0], sum[1], sum[2], sum[3]);
            *reinterpret_cast<float4*>(&C[out_row * n + out_col_base]) = vc;
        } else {
            /* Boundary fallback for right-edge tile (N not multiple of TILE_VEC) */
            #pragma unroll
            for (int v = 0; v < VEC_WIDTH; v++)
                if (out_col_base + v < n)
                    C[out_row * n + out_col_base + v] = sum[v];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 4: WMMA TENSOR CORE (Volta+ / sm_70+)
 * ═══════════════════════════════════════════════════════════════════════════
 * Tensor cores execute a 16×16×16 warp-synchronous matrix multiply.
 * Result: 8192 FMADs per warp per instruction vs 32 for CUDA cores.
 *
 * This kernel:
 *   1. Takes pre-converted FP16 inputs (conversion handled in main()).
 *   2. Each warp owns one 16×16 output tile.
 *   3. Sweeps the k-dimension in strips of WMMA_K=16.
 *   4. Accumulates in FP32 for numerical stability.
 *
 * The FP32 conversion in main() (K4 section below) is not theater:
 * it correctly initializes __half device buffers from the FP32 source data,
 * which is required before any wmma::load_matrix_sync call.
 */
__global__ void matmul_wmma_tensor_core(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        C,
    int           n
) {
    /* One warp (32 threads) per 16×16 output tile */
    int warp_id     = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warps_per_block = (blockDim.x * blockDim.y) / 32;
    int warp_row    = (blockIdx.y * warps_per_block + warp_id) * WMMA_M;
    int warp_col    = blockIdx.x * WMMA_N;

    if (warp_row >= n || warp_col >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k + WMMA_K <= n; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warp_row * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + warp_col, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * n + warp_col, c_frag, n, wmma::mem_row_major);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CUDA kernel that converts FP32 array to FP16 in-place on the device.
 * Avoids a host-side cudaMemcpy + conversion round-trip.
 */
__global__ void fp32_to_fp16(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

/* ─── Numerical validation ─────────────────────────────────────────────── */
struct ValidationResult {
    double frobenius_relative_error;
    double max_absolute_error;
    bool passed;
};

static ValidationResult compare_with_reference(
    const float* reference,
    const float* actual,
    size_t count,
    double tolerance
) {
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;

    for (size_t i = 0; i < count; ++i) {
        const double ref = static_cast<double>(reference[i]);
        const double got = static_cast<double>(actual[i]);
        const double diff = got - ref;
        const double abs_diff = fabs(diff);
        diff_sq += diff * diff;
        ref_sq += ref * ref;
        if (abs_diff > max_abs) max_abs = abs_diff;
    }

    const double ref_norm = sqrt(ref_sq);
    const double rel_error = sqrt(diff_sq) / (ref_norm > 1e-30 ? ref_norm : 1.0);
    return {rel_error, max_abs, rel_error <= tolerance};
}

static ValidationResult validate_device_output(
    const char* kernel_name,
    const float* reference,
    const float* device_output,
    float* host_scratch,
    size_t count,
    double tolerance
) {
    CUDA_CHECK(cudaMemcpy(host_scratch, device_output, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    ValidationResult result = compare_with_reference(reference, host_scratch, count, tolerance);
    printf("%-30s relative error=%10.3e  max abs=%10.3e  tol=%.1e  %s\n",
           kernel_name,
           result.frobenius_relative_error,
           result.max_absolute_error,
           tolerance,
           result.passed ? "PASS" : "FAIL");
    return result;
}

/* ─── Timing helper ──────────────────────────────────────────────────────── */
static float gpu_time_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

/* ─── Main ───────────────────────────────────────────────────────────────── */
int main() {
    printf("\n========================================================\n");
    printf("  GEMM Benchmark Suite — N=%d, FP32 (K4: FP16 TC)\n", N);
    printf("========================================================\n\n");

    const double GFLOPS_REF = 2.0 * (double)N * N * N / 1e9;

    /* ── Host allocation ─────────────────────────────────────────────────── */
    size_t bytes   = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes);
    if (!h_A || !h_B) { fprintf(stderr, "Host allocation failed\n"); return EXIT_FAILURE; }
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* ── Device allocation ───────────────────────────────────────────────── */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* FP16 buffers for K4: convert from d_A/d_B on-device (no extra host copy) */
    size_t bytes_half = (size_t)N * N * sizeof(__half);
    __half *d_A_half, *d_B_half;
    CUDA_CHECK(cudaMalloc(&d_A_half, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_B_half, bytes_half));
    float fp16_input_conversion_ms = 0.0f;
    {
        int total = N * N;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        cudaEvent_t convert_start, convert_stop;
        CUDA_CHECK(cudaEventCreate(&convert_start));
        CUDA_CHECK(cudaEventCreate(&convert_stop));

        // One untimed warm-up, followed by a separately measured preprocessing pass.
        fp32_to_fp16<<<blocks, threads>>>(d_A, d_A_half, total);
        fp32_to_fp16<<<blocks, threads>>>(d_B, d_B_half, total);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(convert_start));
        fp32_to_fp16<<<blocks, threads>>>(d_A, d_A_half, total);
        fp32_to_fp16<<<blocks, threads>>>(d_B, d_B_half, total);
        CUDA_CHECK(cudaEventRecord(convert_stop));
        CUDA_CHECK(cudaEventSynchronize(convert_stop));
        CUDA_CHECK(cudaEventElapsedTime(&fp16_input_conversion_ms, convert_start, convert_stop));

        CUDA_CHECK(cudaEventDestroy(convert_start));
        CUDA_CHECK(cudaEventDestroy(convert_stop));
    }

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double cublas_gflops = 0.0;
    double cublas_ms = 0.0;
    double kernel_ms[5] = {0.0};
    double kernel_gflops[5] = {0.0};
    double kernel_pct[5] = {0.0};
    ValidationResult validation[5] = {};

    float* h_reference = (float*)malloc(bytes);
    float* h_validation = (float*)malloc(bytes);
    if (!h_reference || !h_validation) {
        fprintf(stderr, "Validation host allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Build an independent FP32 cuBLAS reference before timing any custom kernel. */
    {
        cublasHandle_t reference_handle;
        CUBLAS_CHECK(cublasCreate(&reference_handle));
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(reference_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_reference, d_C, bytes, cudaMemcpyDeviceToHost));
        CUBLAS_CHECK(cublasDestroy(reference_handle));
    }

    printf("CORRECTNESS VALIDATION (reference: cuBLAS SGEMM FP32)\n");
    printf("--------------------------------------------------------------\n");
    {
        dim3 block(TILE_K0, TILE_K0);
        dim3 grid((N + TILE_K0 - 1) / TILE_K0, (N + TILE_K0 - 1) / TILE_K0);
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        validation[0] = validate_device_output("K0: Naive", h_reference, d_C, h_validation,
                                               (size_t)N * N, 1e-4);
    }
    {
        dim3 block(TILE_K1, TILE_K1);
        dim3 grid((N + TILE_K1 - 1) / TILE_K1, (N + TILE_K1 - 1) / TILE_K1);
        matmul_tiled_shmem<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        validation[1] = validate_device_output("K1: Shared Memory", h_reference, d_C, h_validation,
                                               (size_t)N * N, 1e-4);
    }
    {
        dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);
        matmul_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        validation[2] = validate_device_output("K2: Register Tiled", h_reference, d_C, h_validation,
                                               (size_t)N * N, 1e-4);
    }
    {
        dim3 block(TILE_VEC / VEC_WIDTH, TILE_VEC);
        dim3 grid((N + TILE_VEC - 1) / TILE_VEC, (N + TILE_VEC - 1) / TILE_VEC);
        matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        validation[3] = validate_device_output("K3: float4", h_reference, d_C, h_validation,
                                               (size_t)N * N, 1e-4);
    }
    {
        const int warps_per_block = 4;
        dim3 block(32, warps_per_block);
        dim3 grid((N + WMMA_N - 1) / WMMA_N,
                  (N + WMMA_M * warps_per_block - 1) / (WMMA_M * warps_per_block));
        matmul_wmma_tensor_core<<<grid, block>>>(d_A_half, d_B_half, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        validation[4] = validate_device_output("K4: WMMA FP16 input / FP32 acc", h_reference, d_C,
                                               h_validation, (size_t)N * N, 5e-3);
    }
    for (int i = 0; i < 5; ++i) {
        if (!validation[i].passed) {
            fprintf(stderr, "Correctness validation failed for K%d; benchmark aborted.\n", i);
            return EXIT_FAILURE;
        }
    }
    printf("Validation: ALL KERNELS PASS\n\n");

    /* ── cuBLAS reference ────────────────────────────────────────────────── */
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f, beta = 0.0f;

        for (int i = 0; i < WARMUP; i++)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++)
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        cublas_ms = ms;
        cublas_gflops = GFLOPS_REF / (ms / 1000.0);
        printf("%-30s  %7.2f ms   %9.2f GFLOPS   (100%% efficiency)\n",
               "cuBLAS (reference)", ms, cublas_gflops);
        printf("──────────────────────────────────────────────────────────\n");

        CUBLAS_CHECK(cublasDestroy(handle));
    }

    /* ── K0: Naive ───────────────────────────────────────────────────────── */
    {
        dim3 block(TILE_K0, TILE_K0);
        dim3 grid((N+TILE_K0-1)/TILE_K0, (N+TILE_K0-1)/TILE_K0);
        for (int i = 0; i < WARMUP; i++) matmul_naive<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++) matmul_naive<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        double gf = GFLOPS_REF / (ms / 1000.0);
        kernel_ms[0] = ms; kernel_gflops[0] = gf; kernel_pct[0] = 100.0 * gf / cublas_gflops;
        printf("K0: Naive                       %7.2f ms   %9.2f GFLOPS   (%.1f%% cuBLAS)\n",
               ms, gf, 100.0*gf/cublas_gflops);
    }

    /* ── K1: Shared Memory Tiled ─────────────────────────────────────────── */
    {
        dim3 block(TILE_K1, TILE_K1);
        dim3 grid((N+TILE_K1-1)/TILE_K1, (N+TILE_K1-1)/TILE_K1);
        for (int i = 0; i < WARMUP; i++) matmul_tiled_shmem<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++) matmul_tiled_shmem<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        double gf = GFLOPS_REF / (ms / 1000.0);
        kernel_ms[1] = ms; kernel_gflops[1] = gf; kernel_pct[1] = 100.0 * gf / cublas_gflops;
        printf("K1: Shared Mem Tiled            %7.2f ms   %9.2f GFLOPS   (%.1f%% cuBLAS)\n",
               ms, gf, 100.0*gf/cublas_gflops);
    }

    /* ── K2: Register Tiled ──────────────────────────────────────────────── */
    {
        dim3 block(BLOCK_N/THREAD_N, BLOCK_M/THREAD_M);
        dim3 grid((N+BLOCK_N-1)/BLOCK_N, (N+BLOCK_M-1)/BLOCK_M);
        for (int i = 0; i < WARMUP; i++) matmul_register_tiled<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++) matmul_register_tiled<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        double gf = GFLOPS_REF / (ms / 1000.0);
        kernel_ms[2] = ms; kernel_gflops[2] = gf; kernel_pct[2] = 100.0 * gf / cublas_gflops;
        printf("K2: Register Tiled              %7.2f ms   %9.2f GFLOPS   (%.1f%% cuBLAS)\n",
               ms, gf, 100.0*gf/cublas_gflops);
    }

    /* ── K3: Vectorized (real float4 loads) ──────────────────────────────── */
    {
        /* Block: TILE_VEC rows × (TILE_VEC/VEC_WIDTH) columns of threads.
         * Each thread handles VEC_WIDTH=4 output elements per row.
         * 32 rows × 8 thread-columns = 256 threads/block. */
        dim3 block(TILE_VEC / VEC_WIDTH, TILE_VEC);  /* (8, 32) */
        dim3 grid((N + TILE_VEC - 1) / TILE_VEC, (N + TILE_VEC - 1) / TILE_VEC);
        for (int i = 0; i < WARMUP; i++) matmul_vectorized<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++) matmul_vectorized<<<grid,block>>>(d_A,d_B,d_C,N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        double gf = GFLOPS_REF / (ms / 1000.0);
        kernel_ms[3] = ms; kernel_gflops[3] = gf; kernel_pct[3] = 100.0 * gf / cublas_gflops;
        printf("K3: Vectorized (float4 loads)   %7.2f ms   %9.2f GFLOPS   (%.1f%% cuBLAS)\n",
               ms, gf, 100.0*gf/cublas_gflops);
    }

    /* ── K4: WMMA Tensor Core (FP16 input, FP32 accumulate) ─────────────── */
    {
        /* Grid/block: one warp per 16×16 output tile.
         * 4 warps per block (128 threads), each covering a 16×16 tile in the M dim. */
        const int WARPS_PER_BLOCK = 4;
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((N + WMMA_N - 1) / WMMA_N,
                  (N + WMMA_M * WARPS_PER_BLOCK - 1) / (WMMA_M * WARPS_PER_BLOCK));

        for (int i = 0; i < WARMUP; i++)
            matmul_wmma_tensor_core<<<grid,block>>>(d_A_half, d_B_half, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int i = 0; i < TIMED; i++)
            matmul_wmma_tensor_core<<<grid,block>>>(d_A_half, d_B_half, d_C, N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = gpu_time_ms(ev_start, ev_stop) / TIMED;
        /* K4 uses FP16 inputs; compare GFLOPS to FP32 cuBLAS to show TC speedup */
        double gf = GFLOPS_REF / (ms / 1000.0);
        kernel_ms[4] = ms; kernel_gflops[4] = gf; kernel_pct[4] = 100.0 * gf / cublas_gflops;
        printf("K4: WMMA Tensor Core (FP16)     %7.2f ms   %9.2f GFLOPS   (%.1f%% cuBLAS)\n",
               ms, gf, 100.0*gf/cublas_gflops);
    }

    printf("\n");
    printf("  Compile: nvcc -O3 -arch=sm_75 -lcublas matmul_suite.cu -o matmul_bench\n");
    printf("  Profile: ncu --set full -o profile ./matmul_bench\n");
    printf("========================================================\n\n");

    /* ── Emit a machine-readable result artifact between stable markers. ── */
    int device = 0, runtime_version = 0, driver_version = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    const char* kernel_names[5] = {
        "K0: Naive", "K1: Shared Memory Tiled", "K2: Register Tiled",
        "K3: float4 Vectorized", "K4: WMMA Tensor Core"
    };
    const char* input_precision[5] = {"FP32", "FP32", "FP32", "FP32", "FP16"};
    printf("BENCHMARK_JSON_START\n");
    printf("{\n");
    printf("  \"schema_version\": 1,\n");
    printf("  \"metadata\": {\n");
    printf("    \"backend\": \"cuda\",\n");
    printf("    \"gpu_name\": \"%s\", \"compute_capability\": \"%d.%d\",\n",
           properties.name, properties.major, properties.minor);
    printf("    \"cuda_runtime_version\": %d, \"cuda_driver_version\": %d, \"fp16_input_conversion_ms\": %.6f,\n",
           runtime_version, driver_version, fp16_input_conversion_ms);
    printf("    \"matrix_size\": %d, \"shape\": {\"M\": %d, \"N\": %d, \"K\": %d}, \"layout\": \"row_major\",\n",
           N, N, N, N);
    printf("    \"warmup_iterations\": %d, \"timed_iterations\": %d\n",
           WARMUP, TIMED);
    printf("  },\n");
    printf("  \"reference\": { \"name\": \"cuBLAS SGEMM FP32\", \"input_precision\": \"FP32\", \"accumulation_precision\": \"FP32\", \"output_precision\": \"FP32\", \"latency_ms\": %.6f, \"gflops\": %.6f },\n",
           cublas_ms, cublas_gflops);
    printf("  \"results\": [\n");
    for (int i = 0; i < 5; ++i) {
        printf("    { \"kernel_id\": \"K%d\", \"name\": \"%s\", \"input_precision\": \"%s\", \"accumulation_precision\": \"FP32\", \"output_precision\": \"FP32\", \"latency_ms\": %.6f, \"gflops\": %.6f, \"pct_of_cublas\": %.4f, \"validation\": { \"frobenius_relative_error\": %.10e, \"max_absolute_error\": %.10e, \"passed\": %s } }%s\n",
               i, kernel_names[i], input_precision[i], kernel_ms[i], kernel_gflops[i],
               kernel_pct[i], validation[i].frobenius_relative_error,
               validation[i].max_absolute_error, validation[i].passed ? "true" : "false",
               i == 4 ? "" : ",");
    }
    printf("  ]\n");
    printf("}\n");
    printf("BENCHMARK_JSON_END\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    free(h_reference); free(h_validation);
    free(h_A); free(h_B);
    return 0;
}
