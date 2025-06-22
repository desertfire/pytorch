#pragma once

#include <sstream>
#include <string>

#include <c10/util/ArrayRef.h>
#include <torch/standalone/core/Device.h>
#include <torch/standalone/core/ScalarType.h>

namespace torch::standalone {

template <class T>
T empty_tensor(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    torch::standalone::ScalarType dtype,
    torch::standalone::Device device,
    int64_t storage_offset) {
  throw std::runtime_error("empty_tensor not implemented");
}

} // namespace torch::standalone
