#include <torch/csrc/inductor/aoti_neutron/cpu/c_shim_cpu.h>

#if defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
// Fallback to standard BLAS C interface
#include <openblas/cblas.h>
#endif

AOTITorchError aoti_torch_cpu_addmm_out(
    AtenTensorHandle out_handle,
    AtenTensorHandle self_handle,
    AtenTensorHandle mat1_handle,
    AtenTensorHandle mat2_handle,
    double beta,
    double alpha) {
  auto out = *reinterpret_cast<torch::native::neutron::SlimTensor*>(out_handle);
  auto self =
      *reinterpret_cast<torch::native::neutron::SlimTensor*>(self_handle);
  auto mat1 =
      *reinterpret_cast<torch::native::neutron::SlimTensor*>(mat1_handle);
  auto mat2 =
      *reinterpret_cast<torch::native::neutron::SlimTensor*>(mat2_handle);

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
      self.dtype() == torch::native::neutron::ScalarType::_float32,
      "Expected Float tensor, got ",
      self.dtype());
  AOTI_TORCH_CHECK(
      mat1.dtype() == torch::native::neutron::ScalarType::_float32,
      "Expected Float tensor, got ",
      mat1.dtype());
  AOTI_TORCH_CHECK(
      mat2.dtype() == torch::native::neutron::ScalarType::_float32,
      "Expected Float tensor, got ",
      mat2.dtype());
  AOTI_TORCH_CHECK(
      out.dtype() == torch::native::neutron::ScalarType::_float32,
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
          static_cast<float*>(out.data_ptr()) + i * n,
          self.data_ptr(),
          n * sizeof(float));
    }
  }

  float* self_ptr = static_cast<float*>(self.data_ptr());
  float* out_ptr = static_cast<float*>(out.data_ptr());
  float* mat1_ptr = static_cast<float*>(mat1.data_ptr());
  float* mat2_ptr = static_cast<float*>(mat2.data_ptr());

  /*

  for (int i = 0; i < m; i++) {
    for (int kk = 0; kk < k; kk++) {
      std::cout << mat1_ptr[i * k + kk] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  for (int kk = 0; kk < k; kk++) {
    for (int j = 0; j < n; j++) {
      std::cout << mat2_ptr[kk * n + j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  for (int kk = 0; kk < 1; kk++) {
    for (int j = 0; j < n; j++) {
      std::cout << self_ptr[kk * n + j] << " ";
    }
    std::cout << std::endl << std::endl;
  }


   for (int i = 0; i < m; i++) {
     for (int j = 0; j < n; j++) {
       float tmp = 0;
       for (int kk = 0; kk < k; kk++) {
         tmp += mat1_ptr[i * k + kk] * mat2_ptr[kk * n + j];
       }
       out_ptr[i * n + j] = tmp * alpha + beta * self_ptr[j];
     }
   }
  */

  cblas_sgemm(
      CblasRowMajor, // Matrix storage order
      CblasNoTrans, // Don't transpose A
      CblasTrans, // Weights are transposed. How do we know the weights are
                  // actually transposed?
      m, // Rows of A and C
      n, // Columns of B and C
      k, // Columns of A and rows of B
      alpha, // Scalar multiplier
      static_cast<const float*>(mat1.data_ptr()), // Matrix A data
      k, // Leading dimension of A
      static_cast<const float*>(mat2.data_ptr()), // Matrix B data
      k, // Leading dimension of B
      beta, // Scalar multiplier for C
      static_cast<float*>(out.data_ptr()), // Matrix C data
      n); // Leading dimension of C

  /*
    // If out wasn't contiguous, copy the buffer back
    if (!out_is_contiguous) {
      out.copy_(out);
    }
  */

  return 0;
}
