#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/resize.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace torch::standalone {

AOTITorchError aoti_torch_cuda_resize_(
    AtenTensorHandle self,
    const int64_t* size,
    int64_t size_len_,
    int32_t* memory_format) {
  try {
    resize_(self, size, size_len_, memory_format);
    return AOTI_TORCH_SUCCESS;
  } catch (const std::exception& e) {
    return AOTI_TORCH_FAILURE;
  }
}

} // namespace torch::standalone

#ifdef __cplusplus
} // extern "C"
#endif
