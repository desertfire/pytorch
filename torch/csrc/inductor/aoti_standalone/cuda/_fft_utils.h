#pragma once
#include <c10/util/DimVector.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <cmath>
#include <vector>

constexpr int64_t cufft_max_ndim = 3;

namespace torch::standalone {

enum class fft_norm_mode : int64_t {
  none = 0, // No normalization (default)
  by_root_n = 1, // Divide by sqrt(signal_size) (ortho)
  by_n = 2 // Divide by signal_size (forward)
};

// TODO: implement mul_:
// https://www.internalfb.com/code/fbsource/[c404cad5db06]/fbcode/caffe2/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp?lines=158-164
inline void mul_(SlimTensor& tensor);

inline double _fft_normalization_scale(
    int64_t normalization,
    const std::vector<int64_t>& sizes,
    const int64_t* dims,
    int64_t dim_len) {
  auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (int64_t i = 0; i < dim_len; i++) {
    signal_numel *= sizes[dims[i]];
  }

  const double scale_denom = (norm == fft_norm_mode::by_root_n)
      ? std::sqrt(static_cast<double>(signal_numel))
      : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

inline void _fft_apply_normalization(
    SlimTensor& slice,
    int64_t normalization,
    const std::vector<int64_t>& sizes,
    const int64_t* dims,
    int64_t dim_len) {
  double scale = _fft_normalization_scale(normalization, sizes, dims, dim_len);
  if (scale == 1.0) {
    return;
  }
  mul_(slice);
}

inline bool use_optimized_cufft_path(const int64_t* dim, int64_t len) {
  if (len > cufft_max_ndim) {
    return false;
  }
  if (len >= 2 && dim[0] == 0 && dim[1] == 1) {
    return false;
  }
  return true;
}

} // namespace torch::standalone
