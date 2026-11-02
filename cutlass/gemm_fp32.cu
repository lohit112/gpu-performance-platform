/*
 * CUTLASS FP32 SIMT GEMM comparison backend.
 *
 * This is an optional adapter using the CUTLASS 2.x device::Gemm interface.
 * It is intentionally separate from the hand-written CUDA executable.
 *
 * Build from the repository root after cloning CUTLASS into third_party/cutlass:
 *   nvcc -O3 -std=c++17 -arch=sm_75 \
 *     -Ithird_party/cutlass/include \
 *     cutlass/gemm_fp32.cu -lcublas -o build/cutlass_gemm_fp32
 *
 * This source is not considered validated until it is compiled and executed
 * on the target GPU. The benchmark output uses the shared schema.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef GEMM_N
#define GEMM_N 4096
#endif
#ifndef WARMUP
#define WARMUP 5
#endif
#ifndef ITERATIONS
#define ITERATIONS 20
#endif
#ifndef CUTLASS_ARCH_TAG
#define CUTLASS_ARCH_TAG cutlass::arch::Sm75
#endif
#ifndef CUTLASS_REVISION
#define CUTLASS_REVISION "not-recorded"
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e)); \
        return EXIT_FAILURE; \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t e = (call); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", \
                     __FILE__, __LINE__, static_cast<int>(e)); \
        return EXIT_FAILURE; \
    } \
} while (0)

int main() {
    using RowMajor = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        float, RowMajor,
        float, RowMajor,
        float, RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        CUTLASS_ARCH_TAG
    >;

    constexpr int M = GEMM_N;
    constexpr int N = GEMM_N;
    constexpr int K = GEMM_N;
    constexpr size_t count = static_cast<size_t>(M) * N;
    constexpr size_t bytes = count * sizeof(float);

    std::vector<float> h_a(count), h_b(count), h_reference(count), h_output(count);
    std::srand(42);
    for (size_t i = 0; i < count; ++i) {
        h_a[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
        h_b[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_d, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    // Row-major A*B is computed through column-major cuBLAS as (B^T*A^T)^T.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_reference.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    Gemm gemm;
    typename Gemm::Arguments arguments(
        {M, N, K}, {d_a, K}, {d_b, N}, {d_c, N}, {d_d, N}, {alpha, beta}
    );
    cutlass::Status status = gemm(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS initial launch failed: %s\n",
                     cutlassGetStatusString(status));
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_d, bytes, cudaMemcpyDeviceToHost));
    double diff_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(h_output[i]) - h_reference[i];
        diff_sq += diff * diff;
        ref_sq += static_cast<double>(h_reference[i]) * h_reference[i];
        if (std::fabs(diff) > max_abs) max_abs = std::fabs(diff);
    }
    const double rel_error = std::sqrt(diff_sq) / std::max(std::sqrt(ref_sq), 1e-30);
    const bool passed = rel_error <= 1e-4;
    std::printf("CUTLASS FP32 correctness: relative_error=%.6e max_abs=%.6e %s\n",
                rel_error, max_abs, passed ? "PASS" : "FAIL");
    if (!passed) return EXIT_FAILURE;

    for (int i = 0; i < WARMUP; ++i) {
        status = gemm(arguments);
        if (status != cutlass::Status::kSuccess) return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        status = gemm(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::fprintf(stderr, "CUTLASS timed launch failed: %s\n",
                         cutlassGetStatusString(status));
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float latency_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&latency_ms, start, stop));
    latency_ms /= ITERATIONS;

    // Time the same FP32 reference operation for a precision-matched comparison.
    for (int i = 0; i < WARMUP; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float reference_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&reference_ms, start, stop));
    reference_ms /= ITERATIONS;

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double cutlass_gflops = flops / (latency_ms * 1.0e6);
    const double reference_gflops = flops / (reference_ms * 1.0e6);

    int device = 0, runtime_version = 0, driver_version = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    std::printf("BENCHMARK_JSON_START\n");
    std::printf("{\n");
    std::printf("  \"schema_version\": 1,\n");
    std::printf("  \"metadata\": {\"backend\":\"cutlass\",\"gpu_name\":\"%s\","
                "\"compute_capability\":\"%d.%d\",\"matrix_size\":%d,"
                "\"warmup_iterations\":%d,\"timed_iterations\":%d,"
                "\"cuda_runtime_version\":%d,\"cuda_driver_version\":%d,"
                "\"cutlass_revision\":\"%s\"},\n",
                properties.name, properties.major, properties.minor,
                M, WARMUP, ITERATIONS, runtime_version, driver_version, CUTLASS_REVISION);
    std::printf("  \"reference\": {\"name\":\"cuBLAS SGEMM FP32\","
                "\"latency_ms\":%.6f,\"gflops\":%.6f},\n",
                reference_ms, reference_gflops);
    std::printf("  \"results\": [{\"kernel_id\":\"CUTLASS_FP32\","
                "\"name\":\"CUTLASS device::Gemm FP32 SIMT\","
                "\"input_precision\":\"FP32\",\"accumulation_precision\":\"FP32\","
                "\"latency_ms\":%.6f,\"gflops\":%.6f,\"pct_of_cublas\":%.4f,"
                "\"validation\":{\"frobenius_relative_error\":%.10e,"
                "\"max_absolute_error\":%.10e,\"passed\":true}}]\n",
                latency_ms, cutlass_gflops,
                100.0 * cutlass_gflops / reference_gflops, rel_error, max_abs);
    std::printf("}\nBENCHMARK_JSON_END\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    return EXIT_SUCCESS;
}
