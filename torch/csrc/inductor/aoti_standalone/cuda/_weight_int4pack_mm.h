#pragma once

#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/cuda/_weight_int4pack_mm.cuh>

#ifdef __cplusplus
extern "C" {
#endif

inline AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    int64_t qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = torch::standalone::_weight_int4pack_mm_cuda<
        torch::standalone::SlimTensor>(
        *self, *mat2, qGroupSize, *qScaleAndZeros);
    *ret0 = new torch::standalone::SlimTensor(std::move(tmp_result));
  });
}

#ifdef __cplusplus
}
#endif
