#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

// addmm_out
AOTITorchError aoti_torch_cpu_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha);

AOTITorchError aoti_torch_cpu_hann_window(
    int64_t window_length,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
    int32_t* pin_memory,
    AtenTensorHandle* ret0);

#ifdef __cplusplus
} // extern "C"
#endif
