#pragma once

#include <c10/core/Scalar.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/utils.h>

namespace torch::standalone {

template <class T>
T _scalar_tensor(
    const c10::Scalar& s,
    c10::ScalarType dtype,
    c10::Device device) {
  c10::IntArrayRef sizes = {};
  c10::IntArrayRef strides = {};

  T result = empty_tensor<T>(sizes, strides, dtype, device, 0);

  result.fill_(s);
  return result;
}

} // namespace torch::standalone
