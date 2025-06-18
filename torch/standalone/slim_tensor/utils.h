#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

// OK to use c10 headers here because their corresponding cpp files will be
// included in the final binary.
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/safe_numerics.h>
#include <torch/standalone/core/Device.h>
#include <torch/standalone/core/ScalarType.h>
#include <torch/standalone/slim_tensor/array_ref.h>

namespace torch::standalone {

#if C10_HAS_BUILTIN_OVERFLOW()
// Helper function for safe numel computation with overflow checks
inline size_t safe_compute_numel(const ArrayRef& sizes) {
  uint64_t n = 1;
  bool overflows = c10::safe_multiplies_u64(sizes, &n);
  constexpr auto numel_max = std::min(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  overflows |= (n > numel_max);
  TORCH_CHECK(!overflows, "numel: integer multiplication overflow");
  return static_cast<size_t>(n);
}

// Helper function for safe nbytes computation with overflow checks
inline size_t safe_compute_nbytes(uint64_t numel, c10::ScalarType dtype) {
  uint64_t element_size = elementSize(dtype);
  uint64_t nbytes;

  bool overflows = c10::mul_overflows(numel, element_size, &nbytes);
  constexpr auto nbytes_max = std::numeric_limits<size_t>::max();
  overflows |= (nbytes > nbytes_max);

  TORCH_CHECK(!overflows, "nbytes: integer multiplication overflow");
  return static_cast<size_t>(nbytes);
}
#endif

inline size_t compute_numel(const ArrayRef& sizes) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return safe_compute_numel(sizes);
#else
  return c10::multiply_integers(sizes);
#endif
}

inline size_t compute_nbytes(const ArrayRef& sizes, c10::ScalarType dtype) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return safe_compute_nbytes(safe_compute_numel(sizes), dtype);
#else
  return compute_numel(sizes) * elementSize(dtype);
#endif
}

inline size_t compute_nbytes(size_t numel, c10::ScalarType dtype) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return safe_compute_nbytes(static_cast<uint64_t>(numel), dtype);
#else
  return numel * elementSize(dtype);
#endif
}

inline int64_t compute_storage_nbytes_contiguous(
    ArrayRef sizes,
    size_t itemsize,
    int64_t storage_offset) {
  int64_t numel = 1;
  for (auto s : sizes) {
    numel *= s;
  }
  return static_cast<int64_t>(itemsize) * (storage_offset + numel);
}

inline int64_t compute_storage_nbytes(
    ArrayRef sizes,
    ArrayRef strides,
    size_t itemsize,
    int64_t storage_offset) {
  if (sizes.empty()) {
    return static_cast<int64_t>(itemsize) * storage_offset;
  }

  int64_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return static_cast<int64_t>(itemsize) * (storage_offset + size);
}

} // namespace torch::standalone
