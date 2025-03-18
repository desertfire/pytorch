#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_libtorch_free/package_loader.h>

namespace torch::inductor {
// A utility class to load a model package compiled by
// AOTInductor in the libtore-free mode and run it. It serves as a test
// harness for the AOTInductor libtorch-free mode, so skipping
// In most of the real C++ inference scenarios under the libtorch-free mode,
// users probably will NOT use this class directly.
class TORCH_API AOTILibtorchFreeLoaderFromAten
    : public aoti::libtorch_free::AOTILibtorchFreeLoader {
 public:
  AOTILibtorchFreeLoaderFromAten() = delete;
  AOTILibtorchFreeLoaderFromAten(const AOTILibtorchFreeLoaderFromAten& other) =
      delete;
  AOTILibtorchFreeLoaderFromAten(AOTILibtorchFreeLoaderFromAten&& other) =
      delete;
  AOTILibtorchFreeLoaderFromAten& operator=(
      const AOTILibtorchFreeLoaderFromAten& other) = delete;
  AOTILibtorchFreeLoaderFromAten& operator=(
      AOTILibtorchFreeLoaderFromAten&& other) = delete;

  AOTILibtorchFreeLoaderFromAten(
      const std::string& model_package_path,
      const std::string& model_name = "model",
      const bool run_single_threaded = false,
      const size_t num_runners = 1);
  ~AOTILibtorchFreeLoaderFromAten() = default;

  // boxed_run is not supported yet
  std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs,
      void* stream_handle = nullptr);
};
} // namespace torch::inductor
#endif
