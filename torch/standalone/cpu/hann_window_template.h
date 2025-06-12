#pragma once
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <array>
#include <cmath>

namespace torch::standalone {

template <class T, class AREF>
T hann_window_template(
    int64_t window_length,
    ScalarType dtype,
    Device device,
    bool periodic = true) {
  const double alpha = 0.5;
  const double beta = 0.5;

  TORCH_CHECK(
      dtype == ScalarType::Float, "hann_window: only float32 supported");
  TORCH_CHECK(
      window_length >= 0, "hann_window: window_length must be non-negative");

  std::array<int64_t, 1> size_buf = {window_length};
  std::array<int64_t, 1> stride_buf = {1};
  AREF sizes = {size_buf.data(), 1};
  AREF strides = {stride_buf.data(), 1};

  T out = empty_tensor<T, AREF>(sizes, strides, dtype, device, /*offset=*/0);
  auto* data = static_cast<float*>(out.data_ptr());

  if (window_length == 0)
    return out;
  if (window_length == 1) {
    data[0] = 1.0f;
    return out;
  }

  if (periodic) {
    window_length++;
  }

  const double omega = 2.0 * M_PI / double(window_length - 1);
  for (int64_t n = 0; n < window_length - 1; n++) {
    data[n] = float(alpha - beta * std::cos(omega * static_cast<double>(n)));
  }
  return out;
}

} // namespace torch::standalone
