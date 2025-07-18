#pragma once

#include <cstdint>
#include <stdexcept>

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

// TODO: implement a proper logging mechanism or simply reuse the c10 logging
#ifndef AOTI_TORCH_CHECK
#define AOTI_TORCH_CHECK(...) ((void)0);
#endif
#ifndef AOTI_TORCH_WARN
#define AOTI_TORCH_WARN(...) ((void)0);
#endif

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

namespace torch::standalone {
inline int64_t maybe_wrap_dim(int64_t dim, int64_t ndim) {
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw std::runtime_error("dimension out of range");
  }
  return dim;
}

} // namespace torch::standalone
