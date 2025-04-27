#pragma once

#include <torch/csrc/inductor/aoti_standalone/c_shim.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cpu_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha);

#ifdef __cplusplus
} // extern "C"
#endif
