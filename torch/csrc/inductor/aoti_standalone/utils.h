#pragma once
#include <cstdint>
#include <stdexcept>

// TODO: move c10 headers to a header-only directory
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <torch/csrc/inductor/aoti_standalone/array_ref.h>

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

#define AOTI_TORCH_CHECK(...) ((void)0);
#define AOTI_TORCH_WARN(...) ((void)0);

#define AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(...)   \
  try {                                                   \
    __VA_ARGS__                                           \
  } catch (const std::exception& e) {                     \
    std::cerr << "Exception in aoti_torch: " << e.what(); \
    return AOTI_TORCH_FAILURE;                            \
  } catch (...) {                                         \
    std::cerr << "Exception in aoti_torch: UNKNOWN";      \
    return AOTI_TORCH_FAILURE;                            \
  }                                                       \
  return AOTI_TORCH_SUCCESS;

namespace torch::native::standalone {
inline size_t compute_numel(const MiniIntArrayRef& sizes) {
  int64_t numel = 1;
  for (auto& s : sizes) {
    numel *= s;
  }
  return numel;
}

inline size_t compute_nbytes(
    const MiniIntArrayRef& sizes,
    c10::ScalarType dtype) {
  return compute_numel(sizes) * c10::elementSize(dtype);
}

inline size_t compute_nbytes(size_t numel, c10::ScalarType dtype) {
  return numel * c10::elementSize(dtype);
}
} // namespace torch::native::standalone
