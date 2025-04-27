#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/c_shim_cuda.h>

namespace torch::native::standalone {
namespace {

void sgemm_cublas(
    SlimTensor& out,
    SlimTensor& A,
    SlimTensor& B,
    SlimTensor& C,
    float beta,
    float alpha) {
  // out = alpha* A @ B + beta * C
  // TODO: check contiguous and tranform if needed

  cublasHandle_t handle{};
  cublasCreate(&handle);

  int m = A.size(0);
  int k = A.size(1);
  int n = B.size(1);

  if (C.data_ptr() != out.data_ptr()) {
    // HACK
    for (int64_t i = 0; i < m; i++) {
      cudaMemcpy(
          static_cast<float*>(out.data_ptr()) + i * n,
          C.data_ptr(),
          n * sizeof(float),
          cudaMemcpyDeviceToDevice);
    }
  }

  cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_T,
      n,
      m,
      k,
      &alpha,
      static_cast<const float*>(B.data_ptr()),
      n,
      static_cast<const float*>(A.data_ptr()),
      k,
      &beta, // Compute mm only; add bias in the next step
      static_cast<float*>(out.data_ptr()),
      n);

  /*
  // Compiling .cu files needs extra work to codecache.py and cpp_builder.py
  __global__ void add_bias_kernel(float* output, const float* bias, int
  batch_size, int out_features) {
      // Calculate global thread ID
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      // Check if within bounds
      if (idx < batch_size * out_features) {
          // Get row and column indices
          int row = idx / out_features;
          int col = idx % out_features;

          // Add bias[col] to output[row][col]
          output[idx] += bias[col];
      }
  }
  */

  cublasDestroy(handle);
}

} // namespace
} // namespace torch::native::standalone

/*
AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  if (self->dim() == 2 && mat2->dim() == 2 &&
      self->dtype() == torch::native::standalone::ScalarType::_float32 &&
      mat2->dtype() == torch::native::standalone::ScalarType::_float32) {
    torch::native::standalone::sgemm_cublas(*out, *self, *mat2, *out, 0.0f, 1.0f);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}
*/

AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {
  if (out->dtype() == torch::native::standalone::ScalarType::_float32 &&
      self->dtype() == torch::native::standalone::ScalarType::_float32 &&
      mat1->dtype() == torch::native::standalone::ScalarType::_float32 &&
      mat2->dtype() == torch::native::standalone::ScalarType::_float32) {
    torch::native::standalone::sgemm_cublas(
        *out, *mat1, *mat2, *self, beta, alpha);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}

#endif // USE_CUDA
