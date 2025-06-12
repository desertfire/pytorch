#pragma once

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/util/Factory.h>

namespace torch::standalone {

template <>
SlimTensor empty_tensor(
    ArrayRef sizes,
    ArrayRef strides,
    ScalarType dtype,
    Device device,

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
