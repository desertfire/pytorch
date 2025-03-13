#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>

#define AOTI_LIBTORCH_FREE
// Need to define AOTI_LIBTORCH_FREE to make sure the AOTI
// runtime picks up the libtorch-free C shim.
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#undef AOTI_LIBTORCH_FREE

namespace torch::inductor {
class TORCH_API AOTLibtorchFreeRunner {
 public:
  AOTLibtorchFreeRunner() = delete;
  AOTLibtorchFreeRunner(const AOTLibtorchFreeRunner& other) = delete;
  AOTLibtorchFreeRunner(AOTLibtorchFreeRunner&& other) = delete;
  AOTLibtorchFreeRunner& operator=(const AOTLibtorchFreeRunner& other) = delete;
  AOTLibtorchFreeRunner& operator=(AOTLibtorchFreeRunner&& other) = delete;

  AOTLibtorchFreeRunner(
      const std::string& model_so_path,
      const std::string& device_str);
  ~AOTLibtorchFreeRunner();

  // boxed_run is not defined until we sort out the ownership story.
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

  std::vector<std::string> get_call_spec();

 private:
  void* model_so_{nullptr};

  decltype(&AOTInductorModelContainerCreateWithDevice) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  decltype(&AOTInductorModelContainerGetCallSpec) get_call_spec_func_{nullptr};

  AOTInductorModelContainerHandle container_handle_{nullptr};
};

// A utility class to load a model package compiled by
// AOTInductor in the libtore-free mode and run it. It serves as a test
// harness for the AOTInductor libtorch-free mode, so skipping
// In most of the real C++ inference scenarios under the libtorch-free mode,
// users probably will NOT use this class directly.
class TORCH_API AOTILibtorchFreeLoader {
 public:
  AOTILibtorchFreeLoader() = delete;
  AOTILibtorchFreeLoader(const AOTILibtorchFreeLoader& other) = delete;
  AOTILibtorchFreeLoader(AOTILibtorchFreeLoader&& other) = delete;
  AOTILibtorchFreeLoader& operator=(const AOTILibtorchFreeLoader& other) =
      delete;
  AOTILibtorchFreeLoader& operator=(AOTILibtorchFreeLoader&& other) = delete;

  AOTILibtorchFreeLoader(
      const std::string& model_package_path,
      const std::string& model_name = "model");
  ~AOTILibtorchFreeLoader();

  // boxed_run is not supported yet till we solve the ownership problem more
  // properly
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

  std::vector<std::string> get_call_spec();

 private:
  std::string temp_dir_;
  std::unique_ptr<AOTLibtorchFreeRunner> runner_;
  std::unordered_map<std::string, std::string> metadata_;
};

/*
using CreateAOTLibtorchFreeRunnerFunc =
    std::unique_ptr<AOTLibtorchFreeRunner> (*)(
        const std::string& model_so_path,
        size_t num_models,
        const std::string& device_str,
        const std::string& bin_dir);

// Return a global map "device name" -> "aoti model runner create function" for
// all registered in AOTI external backends
TORCH_API std::unordered_map<std::string, CreateAOTLibtorchFreeRunnerFunc>&
getAOTLibtorchFreeRunnerRegistry();

// To register a new external backend in AOTI one needs to create an instance of
// this struct. It is not thread-safe. Becase it is expected to be called during
// the initialization of the program.
struct TORCH_API RegisterAOTLibtorchFreeRunner {
  RegisterAOTLibtorchFreeRunner(
      const std::string& name,
      CreateAOTLibtorchFreeRunnerFunc create_aoti_model_runner_fn) {
    getAOTLibtorchFreeRunnerRegistry()[name] = create_aoti_model_runner_fn;
  }
};
*/
} // namespace torch::inductor
#endif
