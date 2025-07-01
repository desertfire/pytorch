#pragma once

#include <torch/standalone/util/ReshapeUtils.h>

#include <c10/util/Optional.h>
#include <optional>
#include <vector>
namespace torch::standalone {

template <typename T>
inline T _reshape(const T& self, c10::IntArrayRef proposed_shape) {
  std::vector<int64_t> final_shape_vec =
      infer_size(proposed_shape, self.numel());
  c10::IntArrayRef final_shape(final_shape_vec);

  // `compute_stride` return the proper strides to use if this
  // `reshape` can be just a view.
  std::optional<std::vector<int64_t>> new_strides_opt =
      compute_stride(self.sizes(), self.strides(), final_shape);

  // create a view if possible
  if (new_strides_opt.has_value()) {
    T result = self; // creates a copy that shares the Storage
    result.as_strided_(
        final_shape, new_strides_opt.value(), self.storage_offset());
    return result;
  }

  // if a view is not possible, create a contiguous clone and reshape that
  T contiguous_clone = self.clone_contiguous();

  // after cloning, the tensor is already contiguous. We just need to update
  // its metadata to reflect the new shape. This is effectively a view of
  // the new contiguous clone
  return contiguous_clone.reshape_as_view(final_shape);
}

} // namespace torch::standalone
