#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/reshape_template.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_reshape(
    AtenTensorHandle self,
    const int64_t* shape,
    int64_t shape_len_,
    AtenTensorHandle* ret0) {
  const torch::standalone::SlimTensor* self_tensor =
      reinterpret_cast<const torch::standalone::SlimTensor*>(self);

  c10::IntArrayRef shape_ref(shape, shape_len_);

  torch::standalone::SlimTensor result_tensor =
      torch::standalone::_reshape<torch::standalone::SlimTensor>(
          *self_tensor, shape_ref);

  *ret0 = reinterpret_cast<AtenTensorHandle>(
      new torch::standalone::SlimTensor(std::move(result_tensor)));
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif
