#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/pad_template.h>

#ifdef __cplusplus
extern "C" {
#endif

using torch::standalone::SlimTensor;

AOTITorchError aoti_torch_cuda_pad(
    AtenTensorHandle self,
    const int64_t* pad,
    int64_t pad_len_,
    const char* mode,
    double* value,
    AtenTensorHandle* ret0) {
  TORCH_CHECK(self != nullptr, "self tensor is null");
  TORCH_CHECK(ret0 != nullptr, "return handle is null");

  const SlimTensor* self_tensor = reinterpret_cast<const SlimTensor*>(self);
  c10::IntArrayRef pad_ref(pad, pad_len_);
  std::string_view mode_sv(mode);
  std::optional<double> value_opt;
  if (value) {
    value_opt = *value;
  }

  SlimTensor result_tensor = _pad(*self_tensor, pad_ref, mode_sv, value_opt);
  *ret0 = reinterpret_cast<AtenTensorHandle>(
      new SlimTensor(std::move(result_tensor)));
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif
