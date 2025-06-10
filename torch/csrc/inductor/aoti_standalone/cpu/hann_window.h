#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/cpu/hann_window_template.h>
#include <torch/standalone/slim_tensor/array_ref.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif
namespace torch::standalone {

AOTITorchError aoti_torch_cpu_hann_window(
    int64_t window_length,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
    int32_t* pin_memory,
    AtenTensorHandle* ret0) {

  SlimTensor tensor = hann_window_template<SlimTensor, ArrayRef>(
      window_length, ScalarType::Float, DeviceType::CPU, /*periodic=*/true);

  *ret0 = reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(tensor)));
  return AOTI_TORCH_SUCCESS;
}
} // namespace torch::standalone

#ifdef __cplusplus
} // extern "C"
#endif
