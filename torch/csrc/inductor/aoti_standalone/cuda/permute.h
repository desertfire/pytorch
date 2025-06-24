#pragma once
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

inline AOTITorchError aoti_torch_cuda_permute(
    AtenTensorHandle self_h,
    const int64_t* dims,
    int64_t dims_len,
    AtenTensorHandle* ret0) {
  torch::standalone::SlimTensor& self = *reinterpret_cast<torch::standalone::SlimTensor*>(self_h);
  c10::IntArrayRef dims_ref(dims, static_cast<size_t>(dims_len));
  torch::standalone::SlimTensor result = self.permute(dims_ref);
  *ret0 = reinterpret_cast<AtenTensorHandle>(new torch::standalone::SlimTensor(std::move(result)));
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_CUDA
