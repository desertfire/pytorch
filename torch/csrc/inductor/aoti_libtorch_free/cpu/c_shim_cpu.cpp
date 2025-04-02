#include <torch/csrc/inductor/aoti_libtorch_free/cpu/c_shim_cpu.h>

#if defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
// Fallback to standard BLAS C interface
#include <openblas/cblas.h>
#endif

extern "C" void sgemm_(
    char* transa,
    char* transb,
    int* m,
    int* n,
    int* k,
    float* alpha,
    float* a,
    int* lda,
    float* b,
    int* ldb,
    float* beta,
    float* c,
    int* ldc);

AOTITorchError aoti_torch_cpu_addmm_out(
    AtenTensorHandle out_handle,
    AtenTensorHandle self_handle,
    AtenTensorHandle mat1_handle,
    AtenTensorHandle mat2_handle,
    double beta,
    double alpha) {
  auto out = *reinterpret_cast<aoti::libtorch_free::SlimTensor*>(out_handle);
  auto self = *reinterpret_cast<aoti::libtorch_free::SlimTensor*>(self_handle);
  auto mat1 = *reinterpret_cast<aoti::libtorch_free::SlimTensor*>(mat1_handle);
  auto mat2 = *reinterpret_cast<aoti::libtorch_free::SlimTensor*>(mat2_handle);

  int64_t m = mat1.size(0);
  int64_t k = mat1.size(1);
  int64_t n = mat2.size(1);

  // Dimension checks
  AOTI_TORCH_CHECK(
      mat2.size(0) == k,
      "mat2 size(0) must match mat1 size(1), got ",
      mat2.size(0),
      " and ",
      k);

  AOTI_TORCH_CHECK(
      self.dim() == 2 && self.size(0) == m && self.size(1) == n,
      "self must be a matrix with dimensions [",
      m,
      ", ",
      n,
      "]");

  AOTI_TORCH_CHECK(
      out.size(0) == m && out.size(1) == n,
      "out must have dimensions [",
      m,
      ", ",
      n,
      "]");

  // Check data type
  AOTI_TORCH_CHECK(
      self.dtype() == aoti::libtorch_free::ScalarType::_float32,
      "Expected Float tensor, got ",
      self.dtype());
  AOTI_TORCH_CHECK(
      mat1.dtype() == aoti::libtorch_free::ScalarType::_float32,
      "Expected Float tensor, got ",
      mat1.dtype());
  AOTI_TORCH_CHECK(
      mat2.dtype() == aoti::libtorch_free::ScalarType::_float32,
      "Expected Float tensor, got ",
      mat2.dtype());
  AOTI_TORCH_CHECK(
      out.dtype() == aoti::libtorch_free::ScalarType::_float32,
      "Expected Float tensor, got ",
      out.dtype());

  // Check if input tensors are contiguous, if not, create contiguous copies

  if (self.data_ptr() != out.data_ptr()) {
    // TODO: need to call expand_as() to broadcast.
    // In general, we need to implement all kinds of view operations.

    // out.copy_(self);
    //  Hack it here
    for (int64_t i = 0; i < m; i++) {
      memcpy(
          (float*)out.data_ptr() + i * n, self.data_ptr(), n * sizeof(float));
    }
  }

  // Get pointers to the raw data
  float* out_ptr = static_cast<float*>(out.data_ptr());
  float* mat1_ptr = static_cast<float*>(mat1.data_ptr());
  float* mat2_ptr = static_cast<float*>(mat2.data_ptr());

  // BLAS parameters
  char transa = 'n'; // No transpose for mat1
  char transb = 'n'; // No transpose for mat2
  int m_int = static_cast<int>(m);
  int n_int = static_cast<int>(n);
  int k_int = static_cast<int>(k);
  float alpha_val = alpha; // Coefficient for mat1 @ mat2
  float beta_val = beta; // Coefficient for self
  int lda = static_cast<int>(k); // Leading dimension of mat1
  int ldb = static_cast<int>(n); // Leading dimension of mat2
  int ldc = static_cast<int>(n); // Leading dimension of out

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float tmp = 0;
      for (int kk = 0; kk < k; kk++) {
        tmp += mat1_ptr[i * m + kk] * mat2_ptr[kk * n + j];
      }
      out_ptr[i * n + j] = tmp * alpha_val + beta_val * out_ptr[i * n + j];
    }
  }

  /*
    // Call SGEMM directly
    sgemm_(
        &transa,
        &transb,
        &m_int,
        &n_int,
        &k_int,
        &alpha_val,
        mat1_ptr,
        &lda,
        mat2_ptr,
        &ldb,
        &beta_val,
        out_ptr,
        &ldc);
        */

  /*
  // If out wasn't contiguous, copy the buffer back
  if (!out_is_contiguous) {
    out.copy_(out);
  }
  */

  return 0;
}
