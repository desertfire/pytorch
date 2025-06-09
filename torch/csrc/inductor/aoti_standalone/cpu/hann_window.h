#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
using torch::standalone::ArrayRef;
using torch::standalone::SlimTensor;

AOTITorchError aoti_torch_cpu_hann_window(
    int64_t window_length,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
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

  int64_t size_buf[1] = {window_length};
  int64_t stride_buf[1] = {1};

  ArrayRef sizes(size_buf, 1, false);
  ArrayRef strides(stride_buf, 1, false);

  SlimTensor* result = new SlimTensor(empty_tensor(
      sizes, strides, c10::ScalarType::Float, c10::DeviceType::CPU, 0));
  float* data = static_cast<float*>(result->data_ptr());

  if (window_length == 0) {
    *ret0 = reinterpret_cast<AtenTensorHandle>(result);
    return 0;
  } else if (window_length == 1) {
    // fill 1.0 with size=1
    data[0] = 1.0f;
    *ret0 = reinterpret_cast<AtenTensorHandle>(result);
    return 0;
  }

  if (periodic) {
    window_length++;
  }

  const double omega = 2.0 * M_PI / static_cast<double>(window_length - 1);
  for (int64_t n = 0; n < window_length - 1; n++) {
    data[n] = static_cast<float>(alpha - beta * std::cos(omega * n));
  }

  *ret0 = reinterpret_cast<AtenTensorHandle>(result);
  return 0;
}


#ifdef __cplusplus
} // extern "C"
#endif
