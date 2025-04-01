#pragma once

#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_libtorch_free/c_shim.h>

#ifdef __cplusplus
extern "C" {
#endif
AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // USE_CUDA