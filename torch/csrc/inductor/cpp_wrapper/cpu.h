#pragma once

#ifdef AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_standalone/cpu/c_shim_cpu.h>
#else

#include <torch/csrc/inductor/cpp_wrapper/common.h>
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cpu.h>
