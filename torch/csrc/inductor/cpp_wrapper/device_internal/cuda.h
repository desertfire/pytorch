#pragma once

#ifdef AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_libtorch_free/cuda/c_shim_cuda.h>
#include <torch/csrc/inductor/aoti_libtorch_free/utils_cuda.h>
#else
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#endif
