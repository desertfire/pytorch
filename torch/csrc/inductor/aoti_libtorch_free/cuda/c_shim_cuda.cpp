#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_libtorch_free/cuda/c_shim_cuda.h>

namespace aoti::libtorch_free {
namespace {

// CUDA kernel for bias addition
__global__ void add_bias(
    float* out,
    const float* bias,
    int batch_size,
    int out_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * out_features) {
    int feature_idx = idx % out_features;
    out[idx] += bias[feature_idx];
  }
}

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
  float zero_beta = 0;
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
      &zero_beta, // Compute mm only; add bias in the next step
      static_cast<float*>(out.data_ptr()),
      n);

  if (beta != 0.0f) {
    dim3 block(256);
    dim3 grid((m * n + block.x - 1) / block.x);
    add_bias<<<grid, block>>>(static_cast<float*>(out.data_ptr()), static_cast<float*>(C.data_ptr()), m, n);
  }

  cublasDestroy(handle);
}

} // namespace
} // namespace aoti::libtorch_free

/*
AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  if (self->dim() == 2 && mat2->dim() == 2 &&
      self->dtype() == aoti::libtorch_free::ScalarType::_float32 &&
      mat2->dtype() == aoti::libtorch_free::ScalarType::_float32) {
    aoti::libtorch_free::sgemm_cublas(*out, *self, *mat2, *out, 0.0f, 1.0f);
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
  if (self->dim() == 2 && mat2->dim() == 2 &&
      self->dtype() == aoti::libtorch_free::ScalarType::_float32 &&
      mat2->dtype() == aoti::libtorch_free::ScalarType::_float32) {
    aoti::libtorch_free::sgemm_cublas(*out_handle, *mat1, *mat2, *self, beta, alpha);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}

#endif // USE_CUDA
