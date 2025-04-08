#pragma once
#include <cstdint>
#include <stdexcept>

#include <torch/csrc/inductor/aoti_neutron/scalar_type.h>
#include <torch/csrc/inductor/aoti_runtime/mini_array_ref.h>

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

#define AOTI_TORCH_CHECK(...) ((void)0);
#define AOTI_TORCH_WARN(...) ((void)0);

using IntArrayRef = torch::aot_inductor::MiniArrayRef<const int64_t>;

namespace torch::neutron {
inline size_t compute_numel(IntArrayRef sizes) {
  int64_t numel = 1;
  for (auto& s : sizes) {
    numel *= s;
  }
  return numel;
}

inline size_t compute_nbytes(IntArrayRef sizes, ScalarType dtype) {
  return compute_numel(sizes) *
      SCALAR_TYPE_TO_BYTESIZE[static_cast<int32_t>(dtype)];
}

} // namespace torch::neutron
