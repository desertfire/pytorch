#pragma once

#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_neutron/c_shim.h>

#ifdef __cplusplus
extern "C" {
#endif
AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha);

AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    int64_t qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // USE_CUDA
