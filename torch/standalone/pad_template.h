#pragma once

#include <vector>

#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

namespace torch::standalone {

template <typename T>
inline T constant_pad_nd(
    const T& self,
    c10::IntArrayRef pad,
    const c10::Scalar& value) {
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even");

  c10::IntArrayRef input_sizes = self.sizes();
  int64_t l_inp = self.dim();
  int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;
  int64_t l_diff = l_inp - l_pad;

  TORCH_CHECK(
      l_pad <= l_inp,
      "Length of pad should be no more than twice the input's dimension.");

  bool all_pads_non_positive = true;
  T c_input = self;
  for (int64_t i = l_diff; i < l_inp; i++) {
    int64_t pad_idx = 2 * (l_inp - i - 1);

    if (pad[pad_idx] < 0) {
      c_input =
          c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
    } else if (pad[pad_idx] != 0) {
      all_pads_non_positive = false;
    }
    if (pad[pad_idx + 1] < 0) {
      c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
    } else if (pad[pad_idx + 1] != 0) {
      all_pads_non_positive = false;
    }
  }

  // if none of the pads are positive we can optimize and just return the result
  // of calling .narrow() on the input
  if (all_pads_non_positive) {
    return c_input.clone_contiguous();
  }

  // calculate the new shape for the output tensor
  std::vector<int64_t> new_shape;
  new_shape.reserve(l_diff);
  for (int64_t i = 0; i < l_diff; i++) {
    new_shape.emplace_back(input_sizes[i]);
  }

  for (int64_t i = 0; i < l_pad; i++) {
    size_t pad_idx = pad.size() - ((i + 1) * 2);
    int64_t pad_l = pad[pad_idx];
    int64_t pad_r = pad[pad_idx + 1];
    size_t new_dim = input_sizes[l_diff + i] + pad_l + pad_r;

    TORCH_CHECK(
        new_dim > 0,
        "The input size ",
        input_sizes[l_diff + i],
        ", plus negative padding ",
        pad[pad_idx],
        " and ",
        pad[pad_idx + 1],
        " resulted in a negative output size, "
        "which is invalid. Check dimension ",
        l_diff + i,
        " of your input.");

    new_shape.emplace_back(new_dim);
  }

  std::vector<int64_t> contig_strides =
      torch::standalone::compute_contiguous_strides(
          c10::IntArrayRef(new_shape));
  T output = create_empty_tensor(
      new_shape, contig_strides, self.dtype(), self.device());
  output.fill_(value);

  // create a view into the center of the output tensor
  T c_output = output;
  for (int64_t i = l_diff; i < l_inp; i++) {
    size_t pad_idx = 2 * (l_inp - i - 1);
    int64_t pad_l = pad[pad_idx];
    int64_t pad_r = pad[pad_idx + 1];

    if (pad_l > 0) {
      c_output = c_output.narrow(i, pad_l, c_output.size(i) - pad_l);
    }
    if (pad_r > 0) {
      c_output = c_output.narrow(i, 0, c_output.size(i) - pad_r);
    }
  }
  // copy the input data into the center view
  c_output.copy_(c_input);
  return output;
}

template <typename T>
inline T _pad(
    const T& self,
    c10::IntArrayRef pad,
    std::string_view mode,
    std::optional<double> value) {
  if (mode == "constant") {
    return constant_pad_nd(self, pad, value.value_or(0.0));
  }
  TORCH_CHECK(
      false,
      "Unsupported padding mode: ",
      mode,
      ". Only constant mode is available.");
}

} // namespace torch::standalone
