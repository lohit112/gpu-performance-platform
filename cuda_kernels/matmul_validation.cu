// Validation-oriented driver for the existing K0-K4 kernels.
// The feature branch includes a guard around the legacy main() in matmul_suite.cu.
#define GPU_PERF_LIBRARY_ONLY
#include "matmul_suite.cu"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
struct ErrorStats { double relative_frobenius; double max_abs; };
struct TimingStats { double median_ms; double min_ms; double max_ms; };

ErrorStats compare_to_reference(const float* output, const float* reference, size_t count) {
    std::vector<float> actual(count), expected(count);
    CUDA_CHECK(cudaMemcpy(actual.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expected.data(), reference, count * sizeof(float), cudaMemcpyDeviceToHost));
    long double diff_sq = 0.0L, ref_sq = 0.0L;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(actual[i]) - expected[i];
        diff_sq += static_cast<long double>(diff) * diff;
        ref_sq += static_cast<long double>(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(diff));
    }
    const double rel = std::sqrt(static_cast<double>(diff_sq)) /
        std::max(std::sqrt(static_cast<double>(ref_sq)), 1e-30);
    return {rel, max_abs};
}

TimingStats measure(const std::function<void()>& launch, int warmup,
                    int iterations, int samples) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> times;
    times.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(static_cast<double>(ms) / iterations);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    const size_t mid = times.size() / 2;
    const double median = (times.size() % 2 == 0)
        ? (times[mid - 1] + times[mid]) / 2.0 : times[mid];
    return {median, times.front(), times.back()};
}

void write_row(std::ofstream& csv, const std::string& name,
               const std::string& reference_name, const std::string& dtype,
               const std::string& gpu, int cc, const TimingStats& timing,
               const ErrorStats& error, double operations, double ref_ms,
               bool passed) {
    const double gflops = operations / (timing.median_ms * 1.0e6);
    const double ref_gflops = operations / (ref_ms * 1.0e6);
    const double speedup = ref_ms / timing.median_ms;
    csv << name << ',' << reference_name << ',' << gpu << ',' << cc << ','
        << N << ',' << dtype << ',' << std::fixed << std::setprecision(6)
        << timing.median_ms << ',' << timing.min_ms << ',' << timing.max_ms << ','
        << gflops << ',' << ref_gflops << ',' << speedup << ','
        << error.relative_frobenius << ',' << error.max_abs << ','
        << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << std::left << std::setw(30) << name << std::right
              << " median=" << timing.median_ms << " ms"
              << " GFLOP/s=" << gflops
              << " relF=" << error.relative_frobenius
              << " maxAbs=" << error.max_abs
              << " correctness=" << (passed ? "PASS" : "FAIL") << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    const std::string output_path = argc > 1 ? argv[1] : "gemm_validation.csv";
    cudaDeviceProp properties{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::cout << "GEMM validation suite; N=" << N
              << "\nGPU=" << properties.name << " compute_capability="
              << properties.major << '.' << properties.minor << '\n'
              << "K0-K3: FP32 vs cuBLAS Sgemm; K4: FP16 inputs / FP32 accumulation "
              << "vs cuBLAS GemmEx. Conversion time is excluded from kernel-only timing.\n";

    const size_t count = static_cast<size_t>(N) * N;
    const size_t bytes = count * sizeof(float);
    std::vector<float> host_a(count), host_b(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        host_a[i] = distribution(rng);
        host_b[i] = distribution(rng);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ref_fp32 = nullptr;
    __half *d_a_half = nullptr, *d_b_half = nullptr;
    float *d_ref_half = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref_fp32, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref_half, bytes));
    CUDA_CHECK(cudaMalloc(&d_a_half, count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b_half, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int conversion_threads = 256;
    const int conversion_blocks = static_cast<int>((count + conversion_threads - 1) / conversion_threads);
    fp32_to_fp16<<<conversion_blocks, conversion_threads>>>(d_a, d_a_half, static_cast<int>(count));
    fp32_to_fp16<<<conversion_blocks, conversion_threads>>>(d_b, d_b_half, static_cast<int>(count));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    const double operations = 2.0 * static_cast<double>(N) * N * N;

    // The input pointers are intentionally B then A to map row-major A*B to
    // cuBLAS's column-major interpretation of the same row-major buffers.
    const auto fp32_ref = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_b, N, d_a, N, &beta, d_ref_fp32, N));
    };
    fp32_ref();
    CUDA_CHECK(cudaDeviceSynchronize());
    const TimingStats fp32_ref_time = measure(fp32_ref, 2, 3, 5);

    const auto fp16_ref = [&]() {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha,
            d_b_half, CUDA_R_16F, N,
            d_a_half, CUDA_R_16F, N,
            &beta, d_ref_half, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    fp16_ref();
    CUDA_CHECK(cudaDeviceSynchronize());
    const TimingStats fp16_ref_time = measure(fp16_ref, 2, 3, 5);

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "Could not open output CSV: " << output_path << '\n';
        return 2;
    }
    csv << "kernel,reference,gpu,compute_capability,n,dtype,median_ms,min_ms,max_ms,"
           "gflops,reference_gflops,speedup_vs_reference,relative_frobenius_error,"
           "max_abs_error,correctness\n";
    std::cout << "cuBLAS FP32 reference median: " << fp32_ref_time.median_ms
              << " ms\ncuBLAS FP16-input/FP32-accumulation reference median: "
              << fp16_ref_time.median_ms << " ms\n";

    bool all_pass = true;
    const auto run_fp32_kernel = [&](const std::string& name, const std::function<void()>& launch) {
        launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        const ErrorStats err = compare_to_reference(d_c, d_ref_fp32, count);
        const bool passed = err.relative_frobenius <= 5e-5 && err.max_abs <= 5e-3;
        // K0 is intentionally naive and can take seconds at N=4096.
        const TimingStats timing = measure(launch, 1, name == "K0_Naive" ? 1 : 5, 5);
        write_row(csv, name, "cuBLAS_Sgemm_FP32", "FP32", properties.name,
                  properties.major * 10 + properties.minor, timing, err,
                  operations, fp32_ref_time.median_ms, passed);
        all_pass = all_pass && passed;
    };

    run_fp32_kernel("K0_Naive", [&]() {
        dim3 block(TILE_K0, TILE_K0);
        dim3 grid((N + TILE_K0 - 1) / TILE_K0, (N + TILE_K0 - 1) / TILE_K0);
        matmul_naive<<<grid, block>>>(d_a, d_b, d_c, N);
    });
    run_fp32_kernel("K1_Shared_Tile", [&]() {
        dim3 block(TILE_K1, TILE_K1);
        dim3 grid((N + TILE_K1 - 1) / TILE_K1, (N + TILE_K1 - 1) / TILE_K1);
        matmul_tiled_shmem<<<grid, block>>>(d_a, d_b, d_c, N);
    });
    run_fp32_kernel("K2_Register_Tile", [&]() {
        dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);
        matmul_register_tiled<<<grid, block>>>(d_a, d_b, d_c, N);
    });
    run_fp32_kernel("K3_Float4_Vectorized", [&]() {
        dim3 block(TILE_VEC / VEC_WIDTH, TILE_VEC);
        dim3 grid((N + TILE_VEC - 1) / TILE_VEC, (N + TILE_VEC - 1) / TILE_VEC);
        matmul_vectorized<<<grid, block>>>(d_a, d_b, d_c, N);
    });

    const auto launch_wmma = [&]() {
        constexpr int warps_per_block = 4;
        dim3 block(32, warps_per_block);
        dim3 grid((N + WMMA_N - 1) / WMMA_N,
                  (N + WMMA_M * warps_per_block - 1) / (WMMA_M * warps_per_block));
        matmul_wmma_tensor_core<<<grid, block>>>(d_a_half, d_b_half, d_c, N);
    };
    launch_wmma(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    const ErrorStats wmma_err = compare_to_reference(d_c, d_ref_half, count);
    const bool wmma_pass = wmma_err.relative_frobenius <= 5e-3 && wmma_err.max_abs <= 5e-2;
    const TimingStats wmma_time = measure(launch_wmma, 1, 5, 5);
    write_row(csv, "K4_WMMA_FP16_FP32_ACCUM", "cuBLAS_GemmEx_FP16_FP32_ACCUM",
              "FP16_INPUT_FP32_ACCUM", properties.name,
              properties.major * 10 + properties.minor, wmma_time, wmma_err,
              operations, fp16_ref_time.median_ms, wmma_pass);
    all_pass = all_pass && wmma_pass;

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ref_fp32)); CUDA_CHECK(cudaFree(d_ref_half));
    CUDA_CHECK(cudaFree(d_a_half)); CUDA_CHECK(cudaFree(d_b_half));
    csv.close();
    std::cout << "Wrote measured results to " << output_path << '\n'
              << (all_pass ? "ALL CORRECTNESS CHECKS PASSED\n" :
                              "CORRECTNESS FAILURE: inspect CSV and kernel output.\n");
    return all_pass ? 0 : 1;
}
