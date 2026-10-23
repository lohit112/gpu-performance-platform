#include <torch/extension.h>

// Implemented in softmax_kernel.cu. Input and output are contiguous float32
// CUDA tensors shaped [rows, columns].
torch::Tensor softmax_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "softmax_cuda",
        &softmax_cuda,
        "Stable row-wise softmax CUDA kernel (float32, 2-D tensors)"
    );
}
