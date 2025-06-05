#include <torch/csrc/inductor/aoti_standalone/cpu/c_shim_cpu.h>

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
  auto out =
      *reinterpret_cast<torch::native::standalone::SlimTensor*>(out_handle);
  auto self =
      *reinterpret_cast<torch::native::standalone::SlimTensor*>(self_handle);
  auto mat1 =
      *reinterpret_cast<torch::native::standalone::SlimTensor*>(mat1_handle);
  auto mat2 =
      *reinterpret_cast<torch::native::standalone::SlimTensor*>(mat2_handle);

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
      self.dtype() == c10::ScalarType::Float,
      "Expected Float tensor, got ",
      self.dtype());
  AOTI_TORCH_CHECK(
      mat1.dtype() == c10::ScalarType::Float,
      "Expected Float tensor, got ",
      mat1.dtype());
  AOTI_TORCH_CHECK(
      mat2.dtype() == c10::ScalarType::Float,
      "Expected Float tensor, got ",
      mat2.dtype());
  AOTI_TORCH_CHECK(
      out.dtype() == c10::ScalarType::Float,
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

AOTITorchError aoti_torch_cpu_hann_window(
    int64_t window_length,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t* device_index,
    int32_t* pin_memory,
    AtenTensorHandle* ret0) {
  // periodic is not passed as a parameter.
  const bool periodic = true;
  const double alpha = 0.5;
  const double beta = 0.5;

  AOTI_TORCH_CHECK(
      window_length >= 0,
      "hann_window: window_length must be greater than or requal to 0")
  // only cpu implementation for hann_window
  AOTI_TORCH_CHECK(*device == 1, "hann_window: CPU only")
  AOTI_TORCH_CHECK(
      *dtype == static_cast<int32_t>(c10::ScalarType::Float),
      "hann_window: float32 only")

  SlimTensor result =
      SlimTensor::empty({window_length}, c10::ScalarType::Float);
  float* data = static_cast<float*>(result.data_ptr());

  if (window_length == 0) {
    *ret0 = reinterpret_cast<AtenTensorHandle>(&result);
    return 0;
  } else if (window_length == 1) {
    // fill 1.0 with size=1
    data[0] = 1.0f;
    *ret0 = reinterpret_cast<AtenTensorHandle>(&result);
    return 0;
  }

  if (periodic) {
    window_length++;
  }

  const double omega = 2.0 * M_PI / static_cst<double>(L - 1);
  for (int64_t n = 0; n < window_length - 1; n++) {
    data[n] = static_cast<float>(alpha - beta * std::cos(omega * n));
  }

  *ret0 = reinterpret_cast<AtenTensorHandle>(&result);
  return 0;
}
