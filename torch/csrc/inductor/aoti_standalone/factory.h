#pragma once

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/util/Factory.h>

namespace torch::standalone {
template <>
torch::standalone::SlimTensor empty_tensor(
    torch::standalone::ArrayRef sizes,
    torch::standalone::ArrayRef strides,
    torch::standalone::ScalarType dtype,
    torch::standalone::Device device,
    int64_t storage_offset) {
  return create_empty_tensor(
      sizes,
      strides,
      dtype,
      device,
      storage_offset,
      true // own_sizes_and_strides
  );
}
} // namespace torch::standalone
