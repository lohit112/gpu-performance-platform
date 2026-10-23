#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {
constexpr int kThreads = 256;

// One block processes one row. Subtracting the row maximum avoids overflow
// in exp(x) and the usual numerical instability of naive softmax.
__global__ void row_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = static_cast<int64_t>(row) * cols;
    __shared__ float partial[kThreads];

    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[base + col]);
    }
    partial[tid] = local_max;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = partial[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float e = expf(input[base + col] - row_max);
        output[base + col] = e;
        local_sum += e;
    }
    partial[tid] = local_sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / partial[0];
    for (int col = tid; col < cols; col += blockDim.x) {
        output[base + col] *= inv_sum;
    }
}
}  // namespace

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "softmax_cuda expects a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "softmax_cuda currently supports float32 input only");
    TORCH_CHECK(input.dim() == 2,
                "softmax_cuda expects a 2-D [rows, columns] tensor");
    TORCH_CHECK(input.size(1) > 0, "softmax_cuda requires at least one column");
    TORCH_CHECK(input.size(0) <= INT_MAX, "row count exceeds supported CUDA grid size");
    TORCH_CHECK(input.size(1) <= INT_MAX, "column count exceeds supported kernel index range");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto x = input.contiguous();
    auto output = torch::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    if (rows == 0) return output;

    row_softmax_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
