#pragma once
#include <torch/csrc/inductor/aoti_standalone/cuda/utils.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <vector>

namespace torch::standalone {

template <class T>
inline void permute_size_stride_estimation(
    const T& self,
    const int64_t* dims,
    int64_t dims_len,
    int64_t* new_sizes,
    int64_t* new_strides) {
  const int64_t ndim = self.dim();
  TORCH_CHECK(
      ndim == dims_len, "permute: dims length must be equal to tensor.dim()")

  const auto old_sizes = self.sizes();
  const auto old_strides = self.strides();

  std::vector<bool> seen_dims(ndim, false);

  for (int64_t i = 0; i < dims_len; i++) {
    int64_t d = torch::standalone::maybe_wrap_dim(dims[i], ndim);
    TORCH_CHECK(!seen_dims[d], "permute: duplicate dims are not allowed");
    seen_dims[d] = true;
    new_sizes[i] = old_sizes[d];
    new_strides[i] = old_strides[d];
  }
}

template <class T, class AREF>
inline T permute_template(
    const T& self,
    const int64_t* dims,
    int64_t dims_len) {
  int64_t* new_sz = new int64_t[dims_len];
  int64_t* new_st = new int64_t[dims_len];
  permute_size_stride_estimation(self, dims, dims_len, new_sz, new_st);

  AREF sz_ref{new_sz, static_cast<size_t>(dims_len)};
  AREF st_ref{new_st, static_cast<size_t>(dims_len)};

  T result = self;
  result.as_strided_(sz_ref, st_ref, self.storage_offset());

  return result;
}

} // namespace torch::standalone
