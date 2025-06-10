#pragma onceAdd commentMore actions

#include <sstream>
#include <string>

#include <torch/standalone/core/Device.h>
#include <torch/standalone/core/ScalarType.h>

namespace torch::standalone {

template <class T, class A>
T empty_tensor(
    A sizes,
    A strides,
    torch::standalone::ScalarType dtype,
    torch::standalone::Device device,
    int64_t storage_offset) {
  throw std::runtime_error("empty_tensor not implemented");
}

} // namespace torch::standalone
