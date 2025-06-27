#pragma once
#include <numeric>
#include <optional>
#include <vector>

#include <c10/util/ArrayRef.h>

#include <torch/standalone/slim_tensor/utils.h>

// calculates the final concrete shape by also filling in at most one '-1'
// dimension.
inline std::vector<int64_t> infer_size(c10::IntArrayRef shape, int64_t numel) {
  int64_t new_size = 1;
  std::optional<int64_t> infer_dim;
  std::vector<int64_t> result_shape;
  result_shape.reserve(shape.size());

  int64_t ndim = static_cast<int64_t>(shape.size());

  for (int64_t dim = 0; dim < ndim; dim++) {
    if (shape[dim] == -1) {
      TORCH_CHECK(!infer_dim.has_value(), "only one dimension can be inferred");
      infer_dim = dim;
      result_shape.push_back(-1); // placeholder
    } else {
      TORCH_CHECK(shape[dim] >= 0, "invalid shape dimension ", shape[dim]);
      new_size *= shape[dim];
      result_shape.push_back(shape[dim]);
    }
  }

  if (infer_dim.has_value()) {
    TORCH_CHECK(
        new_size != 0,
        "cannot reshape tensor of 0 elements into shape with -1");
    TORCH_CHECK(
        numel % new_size == 0, "shape is invalid for input size ", numel);
    result_shape[*infer_dim] = numel / new_size;
  } else {
    TORCH_CHECK(
        numel == new_size, "shape is invalid for input of size ", numel);
  }
  return result_shape;
}

// it determines if a reshape is possible as a view.
// If so, it returns the new strides
// If not, it returns an empty optional
inline std::optional<std::vector<int64_t>> compute_stride(
    c10::IntArrayRef old_sizes,
    c10::IntArrayRef old_strides,
    c10::IntArrayRef new_sizes) {
  if (old_sizes.empty()) {
    return std::vector<int64_t>(new_sizes.size(), -1);
  }

  // NOTE: stride is arbitrary in the numel() == 0 case;
  // to match NumPy behavior we copy the strides if the size matches, otherwise
  // we use the stride as if it were computed via resize.
  // This could perhaps be combined with the below code, but the complexity
  // didn't seem worth it.
  size_t numel = torch::standalone::compute_numel(old_sizes);

  if (numel == 0 && old_sizes == new_sizes) {
    return old_strides.vec();
  }

  if (numel == 0) {
    int64_t new_sizes_len = static_cast<int64_t>(new_sizes.size());
    std::vector<int64_t> new_strides(new_sizes_len);
    for (int64_t i = new_sizes_len - 1; i >= 0; i--) {
      if (i == new_sizes_len - 1) {
        new_strides[i] = 1;
      } else {
        new_strides[i] =
            std::max<int64_t>(new_sizes[i + 1], 1) * new_strides[i + 1];
      }
    }
    return new_strides;
  }

  std::vector<int64_t> new_strides(new_sizes.size());
  int64_t view_d = static_cast<int64_t>(new_sizes.size()) - 1;
  int64_t chunk_base_stride = old_strides.back();
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;

  for (int64_t tensor_d = static_cast<int64_t>(old_sizes.size()) - 1;
       tensor_d >= 0;
       tensor_d--) {
    tensor_numel *= old_sizes[tensor_d];

    bool is_chunk_end = (tensor_d == 0) ||
        (old_sizes[tensor_d - 1] != 1 &&
         old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride);

    if (is_chunk_end) {
      while (view_d >= 0 &&
             (view_numel < tensor_numel || new_sizes[view_d] == 1)) {
        new_strides[view_d] = view_numel * chunk_base_stride;
        view_numel *= new_sizes[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return std::nullopt; // Not viewable
      }
      if (tensor_d > 0) {
        chunk_base_stride = old_strides[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) {
    return std::nullopt; // not viewable
  }
  return new_strides;
}
