#pragma onceAdd commentMore actions

#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/standalone/util/Factory.h>

namespace torch::standalone {

template <>
inline SlimTensor empty_tensor<SlimTensor, ArrayRef>(
    ArrayRef sizes, // NOLINT(performance-unnecessary-value-param)
    ArrayRef strides, // NOLINT(performance-unnecessary-value-param)
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
