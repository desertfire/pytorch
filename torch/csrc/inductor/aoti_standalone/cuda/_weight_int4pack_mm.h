#pragma once

#include <torch/csrc/inductor/aoti_standalone/c_shim.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    int64_t qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0);

#ifdef __cplusplus
}
#endif
