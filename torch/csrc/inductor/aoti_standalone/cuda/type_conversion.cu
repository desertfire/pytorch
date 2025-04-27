#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace torch::native::standalone {
namespace {
// CUDA kernel for converting bfloat16 to float32
__global__ void convertBF16ToFP32Kernel(
    __nv_bfloat16* input,
    float* output,
    size_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = __bfloat162float(input[idx]);
  }
}
}

void cuda_convertBFloat16ToFloat32(void* src, void* dst, size_t numel) {
  // Define grid and block dimensions for the CUDA kernel
  dim3 block_size(256);
  dim3 grid_size((numel + block_size.x - 1) / block_size.x);

  // Launch the conversion kernel
  convertBF16ToFP32Kernel<<<grid_size, block_size>>>(
      static_cast<__nv_bfloat16*>(src), static_cast<float*>(dst), numel);

  // Check for CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "CUDA error: " + std::string(cudaGetErrorString(error)));
  }
}
} // namespace torch::native::standalone
#endif // USE_CUDA
