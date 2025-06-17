#pragma once
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/slim_tensor/array_ref.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif
namespace torch::standalone {

AOTITorchError aoti_torch_cuda_permute(
    AtenTensorHandle self_h,
    const int64_t* dims,
    int64_t dims_len,
    AtenTensorHandle* ret0) {
  SlimTensor& self = *reinterpret_cast<SlimTensor*>(self_h);
  ArrayRef dims_ref(dims, static_cast<size_t>(dims_len));
  SlimTensor result = self.permute(dims_ref);
  *ret0 = reinterpret_cast<AtenTensorHandle>(new SlimTensor(std::move(result)));
  return AOTI_TORCH_SUCCESS;
}

} // namespace torch::standalone

#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_CUDA
