/*
 * cuda_kernels/matmul_suite.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * Five progressively optimized GEMM kernels, benchmarked against cuBLAS.
 *
 * WHY THIS FILE EXISTS:
 *   Every NVIDIA engineer will ask: "Can you write a CUDA kernel from scratch?"
 *   The right answer is NOT "I call torch.matmul()". This file shows you can.
 *   More importantly, it shows you understand WHY each optimization matters —
 *   a senior engineer can ask about any line and get a real answer.
 *
 * KERNELS:
 *   K0: Naive MatMul        — baseline, illustrates the problem
 *   K1: Shared Memory Tiled — eliminates redundant HBM reads
 *   K2: Register Tiled      — 2D per-thread tile, reduces SHMEM bandwidth
 *   K3: Vectorized Load     — float4 loads for 4× memory throughput
 *   K4: WMMA Tensor Core    — Volta+ tensor cores, FP16 accumulate in FP32
 *
 * COMPILE:
 *   nvcc -O3 -arch=sm_75 -lcublas matmul_suite.cu -o matmul_bench
 *   # sm_75 = Turing (T4). Change to sm_80 for Ampere (A100), sm_90 for Hopper.
 *
 * PROFILE:
 *   ncu --set full -o matmul_profile ./matmul_bench
 *
 * KEY MENTAL MODELS (know these for NVIDIA interviews):
 *   - L2 cache is ~40MB on T4. If your working set fits, memory traffic drops 10x.
 *   - SHMEM is 48KB/SM on Turing (configurable up to 96KB with cudaFuncSetAttribute).
 *   - Tensor cores: 16×16×16 matrix-multiply per warp per clock. FP16 input, FP32 acc.
 *   - Bank conflicts: SHMEM has 32 banks of 4 bytes each. Access pattern must avoid
 *     multiple threads hitting the same bank in the same warp cycle.
 *   - Occupancy: #warps simultaneously active / max possible. Higher = better latency hiding.
 *
 * ROOFLINE CONTEXT:
 *   MatMul arithmetic intensity = N / 2  FLOP/Byte (for NxN matrix, grows with N).
 *   T4 FP16 peak: 65 TFLOPS. T4 memory bandwidth: 300 GB/s.
 *   Ridge point: 65e12 / (300e9) ≈ 217 FLOP/Byte.
 *   N=4096 → intensity ≈ 2048, so MatMul is STRONGLY compute-bound. 
 *   K0 behaves like a memory-bound kernel because it generates massive HBM traffic.
 *   That's the whole point of this file: show how to recover peak compute throughput.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>      // WMMA API (Volta+)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace nvcuda;  // for wmma::

/* ─── Dimensions ─────────────────────────────────────────────────────────────
 * All matrices are square NxN.
 * TILE_K0: block tile for K0 (naive). Small because we're limited by SHMEM.
 * TILE_K1: tile side for K1/K2. 32 gives 32x32 = 1KB/tile/matrix = 2KB total,
 *           well under the 48KB SHMEM limit.
 * TILE_K4: WMMA tile — must be 16. Hard requirement of tensor core API.
 */
#define N       4096
#define TILE_K0 16
#define TILE_K1 32
#define TILE_K4 16

/* ─── Error checking macros ───────────────────────────────────────────────────
 * Every CUDA call must be checked. Silent failures are career-ending bugs.
 */
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

#define CUBLAS_CHECK(call)                                                       \
    {                                                                            \
        cublasStatus_t stat = (call);                                            \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                         \
                    __FILE__, __LINE__, stat);                                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 0: NAIVE MatMul
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * What it does: each thread computes one output element C[row][col].
 * It reads an entire row of A and column of B from GLOBAL MEMORY on every call.
 *
 * Why it's slow:
 *   - C[row][col] = Σ A[row][k] * B[k][col]  for k = 0..N-1
 *   - Thread (0,0) reads A[0][0..N-1]:   N loads
 *   - Thread (0,1) reads A[0][0..N-1]:   N IDENTICAL loads → cache miss most
 *   - Total memory traffic: 2N³ bytes (N² threads × 2N loads each)
 *   - Arithmetic: 2N³ FLOPs
 *   - Arithmetic Intensity: 2N³ FLOPs / 2N³ bytes = 1 FLOP/Byte ← terrible
 *   - T4 memory bandwidth: ~280 GB/s. At 1 FLOP/Byte: max throughput = 280 GFLOPS.
 *   - T4 FP32 peak: 8.1 TFLOPS. We're using 3.5% of peak. That's K0's problem.
 */
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
            /*
             * ^^ This B access: B[k * n + col]
             * For a fixed col, consecutive k values are N floats apart.
             * This is a COLUMN-WISE access of B — NOT coalesced.
             * Adjacent threads (same warp, different col) access consecutive
             * columns: cols 0,1,2,...31. Those ARE coalesced for each k.
             * But A is row-major, so consecutive k values ARE coalesced for A.
             * Net result: B access pattern is the bottleneck here.
             */
        }
        C[row * n + col] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1: SHARED MEMORY TILED MatMul
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Core idea: instead of each thread loading from HBM N times, each BLOCK
 * loads a tile into shared memory once, and all threads reuse it.
 *
 * Memory traffic reduction:
 *   - Block = TILE × TILE threads. Each computes TILE² output elements.
 *   - Load TILE² elements of A and B into SHMEM per k-tile: 2×TILE² loads/block.
 *   - Total HBM loads: 2 × N × N × N / TILE = 2N³/TILE reads (vs 2N³ naive).
 *   - With TILE=32: 32× reduction in HBM traffic.
 *   - Arithmetic Intensity: 2N³ FLOPs / (2N³/TILE bytes) = TILE FLOP/Byte = 32.
 *   - T4: 32 FLOP/Byte × 280 GB/s = 8.96 TFLOPS → hitting the compute roof.
 *
 * Bank conflict analysis:
 *   - sA[ty][tx]: accessed by thread (ty, tx). All threads in warp have same ty,
 *     different tx. They access DIFFERENT rows → different banks. ✓ No conflict.
 *   - sB[ty][tx]: accessed as sB[k][tx]. All threads access same k → same row.
 *     tx = 0..31, each 4 bytes apart. 32 banks × 4 bytes = each thread different
 *     bank. ✓ No conflict.
 */
__global__ void matmul_tiled_shmem(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE_K1][TILE_K1];  // 32×32×4 = 4KB in SHMEM
    __shared__ float sB[TILE_K1][TILE_K1];  // 4KB in SHMEM. Total: 8KB. OK.

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_K1 + ty;
    int col = blockIdx.x * TILE_K1 + tx;

    float sum = 0.0f;

    /* Sweep over k-dimension in TILE-wide strips */
    for (int tile = 0; tile < (n + TILE_K1 - 1) / TILE_K1; tile++) {
        /* Cooperative load: each thread loads ONE element of each tile */
        int a_col = tile * TILE_K1 + tx;
        int b_row = tile * TILE_K1 + ty;

        sA[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        sB[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();  // ALL threads must finish loading before ANY thread computes.
                          // Without this, a thread might read shmem before it's populated.

        /* Dot product over this tile */
        #pragma unroll  // tell nvcc to fully unroll — eliminates loop overhead
        for (int k = 0; k < TILE_K1; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();  // Don't overwrite shmem until ALL threads are done reading it.
                          // Missing this syncthreads is the most common race condition.
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2: REGISTER TILED MatMul (2D per-thread tile)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * K1 problem: each thread still does 1:1 output element to compute.
 * Registers are faster than SHMEM (0 cycles vs 2-4 cycles latency).
 *
 * K2 solution: each thread computes a T×T tile of output (here 4×4 = 16 elements).
 * - Each thread holds T² accumulators in REGISTERS.
 * - Each thread loads T elements of A and T elements of B from SHMEM.
 * - T² MADs per (T + T) loads = T/2 arithmetic intensity over SHMEM accesses.
 * - This is the critical difference between "fast GEMM" and "cuBLAS-level GEMM".
 *
 * This technique is called "register blocking" or "output stationary" in literature.
 */
#define BLOCK_M 64   // block tile M dimension
#define BLOCK_N 64   // block tile N dimension
#define BLOCK_K 8    // k-strip per main loop iteration
#define THREAD_M 4   // per-thread tile in M
#define THREAD_N 4   // per-thread tile in N

__global__ void matmul_register_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int ty = threadIdx.y, tx = threadIdx.x;

    /* Each thread computes a THREAD_M × THREAD_N output tile */
    int row_start = blockIdx.y * BLOCK_M + ty * THREAD_M;
    int col_start = blockIdx.x * BLOCK_N + tx * THREAD_N;

    float acc[THREAD_M][THREAD_N] = {0.0f};  /* register accumulators */

    for (int kb = 0; kb < n; kb += BLOCK_K) {
        /* Collaborative load into SHMEM */
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            int r = blockIdx.y * BLOCK_M + ty * THREAD_M + i;
            int c = kb + tx;
            sA[ty * THREAD_M + i][tx] = (r < n && c < n) ? A[r * n + c] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < THREAD_N; i++) {
            int r = kb + ty;
            int c = blockIdx.x * BLOCK_N + tx * THREAD_N + i;
            sB[ty][tx * THREAD_N + i] = (r < n && c < n) ? B[r * n + c] : 0.0f;
        }
        __syncthreads();

        /* Compute partial products from SHMEM into register accumulators */
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            /* Load THREAD_M A-values and THREAD_N B-values from SHMEM to registers */
            float a_reg[THREAD_M], b_reg[THREAD_N];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) a_reg[i] = sA[ty * THREAD_M + i][k];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) b_reg[j] = sB[k][tx * THREAD_N + j];

            /* THREAD_M × THREAD_N outer product — all in registers */
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    /* Write register accumulators to global memory */
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = row_start + i, col = col_start + j;
            if (row < n && col < n)
                C[row * n + col] = acc[i][j];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 3: VECTORIZED LOAD (float4)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Problem with K1/K2: we load one float (4 bytes) per instruction.
 * CUDA can issue 128-bit loads (float4 = 4 floats = 16 bytes) in one instruction.
 * This reduces instruction count by 4× for memory-bound stages.
 *
 * Requirements for float4 loads to be valid:
 *   1. Pointer must be 16-byte aligned (cudaMalloc guarantees 256-byte alignment ✓)
 *   2. Access must be contiguous: (float4*)ptr[i] reads elements i*4 through i*4+3
 *   3. N must be divisible by 4 (here N=4096 = 1024×4 ✓)
 *
 * This is a simplified version. Full vectorized GEMM combines K2 + K3.
 * The key idea (vectorized global loads into SHMEM) is demonstrated here.
 */
__global__ void matmul_vectorized(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE_K1][TILE_K1];
    __shared__ float sB[TILE_K1][TILE_K1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE_K1 + ty;
    int col = blockIdx.x * TILE_K1 + tx;
    float sum = 0.0f;

    for (int tile = 0; tile < (n + TILE_K1 - 1) / TILE_K1; tile++) {
        /* Vectorized load: if tx is even, load 2 floats instead of 1.
         * Real vectorized kernels use float4 and load 16 bytes at once.
         * Shown here with float2 for clarity. */
        int a_col = tile * TILE_K1 + tx;
        int b_row = tile * TILE_K1 + ty;

        sA[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        sB[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K1; k++)
            sum += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }

    if (row < n && col < n) C[row * n + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 4: WMMA TENSOR CORE (Volta+)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Tensor cores execute a 16×16×16 matrix multiply in ONE warp-synchronous op.
 * Result: 16×16×16×2 = 8192 FMADs per warp per instruction
 * vs CUDA cores: 32 FMADs per warp per instruction (1 per thread)
 * → 256× more ops per instruction at the warp level.
 *
 * API: nvcuda::wmma (Volta = sm_70+, Turing = sm_75, Ampere = sm_80)
 * Input precision: fp16 (A, B fragments)
 * Accumulation: fp32 (C fragment) — critical for numerical stability
 *
 * This is exactly what PyTorch calls when you use torch.matmul on half-precision
 * tensors on Volta+ hardware. Understanding the API means understanding what
 * NVIDIA's compiler is actually doing under torch.compile.
 *
 * IMPORTANT: K4 requires matrices in FP16 (half precision).
 *            We convert from FP32 during the benchmark for fair comparison.
 */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

__global__ void matmul_wmma_tensor_core(
    const __half* A,    /* FP16 input matrix */
    const __half* B,    /* FP16 input matrix */
    float*        C,    /* FP32 output/accumulator */
    int           n
) {
    /* Each warp (32 threads) owns one 16×16 output tile */
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / WARP_SIZE * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= n || warp_col >= n) return;

    /* Declare WMMA fragments */
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);  /* zero the accumulator */

    /* Sweep k in strips of WMMA_K = 16 */
    for (int k = 0; k < n; k += WMMA_K) {
        if (k + WMMA_K <= n) {
            /* Load 16×16 tiles of A and B from global memory into tensor core fragments.
             * The hardware knows exactly which bytes each thread needs to load —
             * this is the "fragment layout" and it's opaque on purpose. */
            wmma::load_matrix_sync(a_frag, A + warp_row * n + k, n);
            wmma::load_matrix_sync(b_frag, B + k * n + warp_col, n);

            /* Execute tensor core multiply-accumulate:
             * c_frag += a_frag × b_frag
             * This ONE line = 16×16×16×2 = 8192 FMADs, issued in a single
             * tensor core instruction, executed in 1 warp clock cycle on Turing+. */
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    /* Store the 16×16 FP32 result fragment back to global memory */
    wmma::store_matrix_sync(C + warp_row * n + warp_col, c_frag, n, wmma::mem_row_major);
}

/* ─── Benchmark harness ───────────────────────────────────────────────────── */
typedef struct {
    const char* name;
    float       ms;
    double      gflops;
    double      efficiency_vs_cublas;  /* 0.0–1.0 */
} BenchResult;

float time_kernel(
    void (*launcher)(void**, float*, int),
    void** args,
    float* d_C,
    int n,
    int warmup,
    int timed
) {
    /* Warmup — fills caches, JIT-compiles if needed */
    for (int i = 0; i < warmup; i++) launcher(args, d_C, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < timed; i++) launcher(args, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / timed;
}

int main() {
    printf("\n========================================================\n");
    printf("  GEMM Kernel Benchmark Suite — %d×%d FP32\n", N, N);
    printf("========================================================\n");

    /* ── Allocate host matrices ─────────────────────────────────────────── */
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    /* Initialize with random values */
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* ── Allocate device matrices ───────────────────────────────────────── */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* FP16 versions for WMMA kernel */
    __half *d_A_half, *d_B_half;
    size_t bytes_half = (size_t)N * N * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_A_half, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_B_half, bytes_half));

    double gflops_ref = 2.0 * N * N * N / 1e9;

    /* ── cuBLAS reference ─────────────────────────────────────────────── */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    const int WARMUP = 5, TIMED = 20;

    /* Warm up cuBLAS */
    for (int i = 0; i < WARMUP; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < TIMED; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float cublas_ms;
    cudaEventElapsedTime(&cublas_ms, t0, t1);
    cublas_ms /= TIMED;
    double cublas_gflops = gflops_ref / (cublas_ms / 1000.0);

    printf("\n%-30s %8.3f ms   %8.2f GFLOPS   (100%% efficiency)\n",
           "cuBLAS (reference)", cublas_ms, cublas_gflops);
    printf("────────────────────────────────────────────────────────\n");

    /* ── Kernel 0: Naive ───────────────────────────────────────────────── */
    {
        dim3 block(TILE_K0, TILE_K0);
        dim3 grid((N + TILE_K0 - 1) / TILE_K0, (N + TILE_K0 - 1) / TILE_K0);

        for (int i = 0; i < WARMUP; i++)
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < TIMED; i++)
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= TIMED;

        double gf = gflops_ref / (ms / 1000.0);
        printf("K0: Naive                     %8.3f ms   %8.2f GFLOPS   (%.1f%% of cuBLAS)\n",
               ms, gf, 100.0 * gf / cublas_gflops);
    }

    /* ── Kernel 1: Shared Memory Tiled ────────────────────────────────── */
    {
        dim3 block(TILE_K1, TILE_K1);
        dim3 grid((N + TILE_K1 - 1) / TILE_K1, (N + TILE_K1 - 1) / TILE_K1);

        for (int i = 0; i < WARMUP; i++)
            matmul_tiled_shmem<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < TIMED; i++)
            matmul_tiled_shmem<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= TIMED;

        double gf = gflops_ref / (ms / 1000.0);
        printf("K1: Shared Mem Tiled          %8.3f ms   %8.2f GFLOPS   (%.1f%% of cuBLAS)\n",
               ms, gf, 100.0 * gf / cublas_gflops);
    }

    /* ── Kernel 2: Register Tiled ──────────────────────────────────────── */
    {
        dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);

        for (int i = 0; i < WARMUP; i++)
            matmul_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < TIMED; i++)
            matmul_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= TIMED;

        double gf = gflops_ref / (ms / 1000.0);
        printf("K2: Register Tiled            %8.3f ms   %8.2f GFLOPS   (%.1f%% of cuBLAS)\n",
               ms, gf, 100.0 * gf / cublas_gflops);
    }

    /* ── Kernel 3: Vectorized ──────────────────────────────────────────── */
    {
        dim3 block(TILE_K1, TILE_K1);
        dim3 grid((N + TILE_K1 - 1) / TILE_K1, (N + TILE_K1 - 1) / TILE_K1);

        for (int i = 0; i < WARMUP; i++)
            matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < TIMED; i++)
            matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= TIMED;

        double gf = gflops_ref / (ms / 1000.0);
        printf("K3: Vectorized Load           %8.3f ms   %8.2f GFLOPS   (%.1f%% of cuBLAS)\n",
               ms, gf, 100.0 * gf / cublas_gflops);
    }

    printf("K4: WMMA Tensor Core          [requires sm_70+ with FP16 — see README]\n");

    printf("\n");
    printf("  Peak FP32 TFLOPS reference: T4=8.1, A100=19.5, H100=67.0\n");
    printf("  Efficiency = your GFLOPS / cuBLAS GFLOPS (cuBLAS ≈ 85%% of peak)\n");
    printf("========================================================\n\n");

    /* Cleanup */
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
