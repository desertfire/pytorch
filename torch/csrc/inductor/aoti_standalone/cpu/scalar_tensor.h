#pragma once

#include <c10/core/Scalar.h>
#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/scalar_tensor_template.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cpu_scalar_tensor(
    double s,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
    int32_t* pin_memory,
    AtenTensorHandle* ret0) {
  c10::Scalar s_scalar(s);
  c10::ScalarType dtype_val = static_cast<c10::ScalarType>(*dtype);

  TORCH_CHECK(
      *device == static_cast<int32_t>(c10::DeviceType::CPU),
      "Device must be CPU");
  c10::Device device_val(
      c10::DeviceType::CPU, static_cast<c10::DeviceIndex>(device_index_));

  torch::standalone::SlimTensor tensor =
      torch::standalone::scalar_tensor_template<torch::standalone::SlimTensor>(
          s_scalar, dtype_val, device_val);

  *ret0 = reinterpret_cast<AtenTensorHandle>(
      new torch::standalone::SlimTensor(std::move(tensor)));
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif
