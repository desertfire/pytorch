#pragma once

#include <torch/csrc/inductor/aoti_neutron/c_shim.h>

namespace torch::neutron {

template <typename T>
T _weight_int4pack_mm_cuda(
    const T& A,
    const T& B,
    int64_t qGroupSize,
    const T& qScaleAndZeros);

// input is [n][k / 2] (uint8 dtype)
// output is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles / 2] (int32 dtype)
template <typename T>
T _convert_weight_to_int4pack_cuda(const T& in, int64_t innerKTiles);

} // namespace torch::neutron

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
