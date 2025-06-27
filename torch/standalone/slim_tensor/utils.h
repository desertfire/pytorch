#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

// OK to use c10 headers here because their corresponding cpp files will be
// included in the final binary.
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/safe_numerics.h>
#include <torch/standalone/core/Device.h>
#include <torch/standalone/core/ScalarType.h>

namespace torch::standalone {

#if C10_HAS_BUILTIN_OVERFLOW()
// Helper function for safe numel computation with overflow checks
inline size_t safe_compute_numel(c10::IntArrayRef sizes) {
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
  uint64_t nbytes = 0;

  bool overflows = c10::mul_overflows(numel, element_size, &nbytes);
  constexpr auto nbytes_max = std::numeric_limits<size_t>::max();
  overflows |= (nbytes > nbytes_max);

  TORCH_CHECK(!overflows, "nbytes: integer multiplication overflow");
  return static_cast<size_t>(nbytes);
}

// Helper function for safe storage nbytes computation with overflow checks
inline int64_t safe_compute_storage_nbytes(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    size_t itemsize,
    int64_t storage_offset) {
  if (sizes.empty()) {
    uint64_t result = 0;
    bool overflows = c10::mul_overflows(
        static_cast<uint64_t>(itemsize),
        static_cast<uint64_t>(storage_offset),
        &result);
    constexpr auto max_val =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    overflows |= (result > max_val);
    TORCH_CHECK(!overflows, "storage_nbytes: integer multiplication overflow");
    return static_cast<int64_t>(result);
  }

  uint64_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }

    uint64_t stride_contribution = 0;
    bool overflows = c10::mul_overflows(
        static_cast<uint64_t>(strides[i]),
        static_cast<uint64_t>(sizes[i] - 1),
        &stride_contribution);
    TORCH_CHECK(!overflows, "storage_nbytes: stride computation overflow");

    overflows = c10::add_overflows(size, stride_contribution, &size);
    TORCH_CHECK(!overflows, "storage_nbytes: size accumulation overflow");
  }

  uint64_t total_size = 0;
  bool overflows = c10::add_overflows(
      size, static_cast<uint64_t>(storage_offset), &total_size);
  TORCH_CHECK(!overflows, "storage_nbytes: storage_offset addition overflow");

  uint64_t result = 0;
  overflows =
      c10::mul_overflows(static_cast<uint64_t>(itemsize), total_size, &result);
  constexpr auto max_val =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  overflows |= (result > max_val);
  TORCH_CHECK(!overflows, "storage_nbytes: final multiplication overflow");

  return static_cast<int64_t>(result);
}
#endif

inline size_t compute_numel(c10::IntArrayRef sizes) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return safe_compute_numel(sizes);
#else
  return c10::multiply_integers(sizes);
#endif
}

inline size_t compute_nbytes(c10::IntArrayRef sizes, c10::ScalarType dtype) {
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
    c10::IntArrayRef sizes,
    size_t itemsize,
    int64_t storage_offset) {
  int64_t numel = static_cast<int64_t>(compute_numel(sizes));
  return static_cast<int64_t>(itemsize) * (storage_offset + numel);
}

inline int64_t compute_storage_nbytes(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    size_t itemsize,
    int64_t storage_offset) {
#if C10_HAS_BUILTIN_OVERFLOW()
  return safe_compute_storage_nbytes(sizes, strides, itemsize, storage_offset);
#else
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
#endif
}

template <typename scalar_t>
void elementwise_copy(
    scalar_t* dst_data,
    const scalar_t* src_data,
    size_t numel,
    size_t ndim,
    c10::IntArrayRef sizes,
    c10::IntArrayRef src_strides) {
  if (numel == 0) {
    return;
  }

  std::vector<int64_t> counter(ndim, 0);
  for (size_t i = 0; i < numel; i++) {
    int64_t src_offset = 0;
    for (size_t d = 0; d < ndim; d++) {
      src_offset += counter[d] * src_strides[d];
    }

    // the destination is contiguous so its offset is simply the element index
    // 'i'.
    dst_data[i] = src_data[src_offset];

    // increment the multi-dimensional counter to find the next logical element
    int64_t current_dim = static_cast<int64_t>(ndim) - 1;
    while (current_dim >= 0) {
      counter[current_dim]++;
      if (counter[current_dim] < sizes[current_dim]) {
        break;
      }
      counter[current_dim] = 0;
      current_dim--;
    }
  }
}

} // namespace torch::standalone
