#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/standalone/resize_template.h>

#ifdef __cplusplus
extern "C" {
#endif

namespace torch::standalone {

AOTITorchError aoti_torch_cpu_resize_(
    AtenTensorHandle self,
    const int64_t* size,
    int64_t size_len_,
    int32_t* memory_format) {
  SlimTensor* tensor = reinterpret_cast<SlimTensor*>(self);
  c10::IntArrayRef size_ref(size, size_len_);
  std::optional<c10::MemoryFormat> optional_memory_format;
  if (memory_format) {
    optional_memory_format = static_cast<c10::MemoryFormat>(*memory_format);
  }
  _resize_(*tensor, size_ref, optional_memory_format);
  return AOTI_TORCH_SUCCESS;
}

} // namespace torch::standalone

#ifdef __cplusplus
} // extern "C"
#endif
