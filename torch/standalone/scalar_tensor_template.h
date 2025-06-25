#pragma once

#include <c10/core/Scalar.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/utils.h>

namespace torch::standalone {

template <typename T>
inline void scalar_fill(T& tensor, const c10::Scalar& value) {
  if (tensor.numel() != 1) {
    TORCH_CHECK(false, "fill_scalar is only for tensors with 1 element");
  }

  auto fill_value = [&](auto typed_value) {
    using SType = decltype(typed_value);
    *static_cast<SType*>(tensor.data_ptr()) = typed_value;
  };

  switch (tensor.dtype()) {
    case c10::ScalarType::Double:
      fill_value(value.to<double>());
      break;
    case c10::ScalarType::Float:
      fill_value(value.to<float>());
      break;
    case c10::ScalarType::Long:
      fill_value(value.to<int64_t>());
      break;
    case c10::ScalarType::Int:
      fill_value(value.to<int32_t>());
      break;
    case c10::ScalarType::Short:
      fill_value(value.to<int16_t>());
      break;
    case c10::ScalarType::Char:
      fill_value(value.to<int8_t>());
      break;
    case c10::ScalarType::Byte:
      fill_value(value.to<uint8_t>());
      break;
    case c10::ScalarType::Bool:
      fill_value(value.to<bool>());
      break;
    default:
      TORCH_CHECK(false, "scalar_fill: Unsupported dtype");
  }
}

template <class T>
T scalar_tensor_template(
    const c10::Scalar& s,
    c10::ScalarType dtype,
    c10::Device device) {
  c10::IntArrayRef sizes = {};
  c10::IntArrayRef strides = {};

  T result = empty_tensor<T>(sizes, strides, dtype, device, 0);

  scalar_fill(result, s);
  return result;
}

} // namespace torch::standalone
