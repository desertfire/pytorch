#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_libtorch_free/cuda/c_shim_cuda.h>

namespace aoti::libtorch_free {
namespace {
void matmul_float32_cublas(SlimTensor& A, SlimTensor& B, SlimTensor& C) {
  // TODO: check contiguous and tranform if needed
  cublasHandle_t handle{};
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;
  int m = A.size(0);
  int k = A.size(1);
  int n = B.size(1);
  cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_T, // Transpose A and B
      n,
      m,
      k,
      &alpha,
      static_cast<const float*>(B.data_ptr()),
      n,
      static_cast<const float*>(A.data_ptr()),
      k,
      &beta,
      static_cast<float*>(C.data_ptr()),
      n);

  cublasDestroy(handle);
}
} // namespace
} // namespace aoti::libtorch_free

AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  if (self->dim() == 2 && mat2->dim() == 2 &&
      self->dtype() == aoti::libtorch_free::ScalarType::_float32 &&
      mat2->dtype() == aoti::libtorch_free::ScalarType::_float32) {
    aoti::libtorch_free::matmul_float32_cublas(*self, *mat2, *out);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}

#endif // USE_CUDA