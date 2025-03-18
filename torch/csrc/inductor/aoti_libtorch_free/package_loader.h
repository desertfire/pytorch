#pragma once

#include <string>
#include <unordered_map>

#ifdef AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#else
// AOTI_LIBTORCH_FREE is not defined, so we need to define
// AOTI_LIBTORCH_FREE to make sure the AOTI runtime picks up
// the libtorch-free C shim. This is needed because we want
// to be able to test the libtorch-free mode in pytorch.
#define AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#undef AOTI_LIBTORCH_FREE
#endif // AOTI_LIBTORCH_FREE

namespace aoti::libtorch_free {
class AOTLibtorchFreeRunner {
 public:
  AOTLibtorchFreeRunner() = delete;
  AOTLibtorchFreeRunner(const AOTLibtorchFreeRunner& other) = delete;
  AOTLibtorchFreeRunner(AOTLibtorchFreeRunner&& other) = delete;
  AOTLibtorchFreeRunner& operator=(const AOTLibtorchFreeRunner& other) = delete;
  AOTLibtorchFreeRunner& operator=(AOTLibtorchFreeRunner&& other) = delete;

  AOTLibtorchFreeRunner(
      const std::string& model_so_path,
      const std::string& device_str,
      const std::string& cubin_dir,
      const bool run_single_threaded = false,
      const size_t num_runners = 1);

  ~AOTLibtorchFreeRunner();

  // boxed_run is not defined until we sort out the ownership story.
  std::vector<aoti::libtorch_free::SlimTensor> run(
      const std::vector<aoti::libtorch_free::SlimTensor>& inputs,
      void* stream_handle = nullptr);

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
class AOTILibtorchFreeLoader {
 public:
  AOTILibtorchFreeLoader() = delete;
  AOTILibtorchFreeLoader(const AOTILibtorchFreeLoader& other) = delete;
  AOTILibtorchFreeLoader(AOTILibtorchFreeLoader&& other) = delete;
  AOTILibtorchFreeLoader& operator=(const AOTILibtorchFreeLoader& other) =
      delete;
  AOTILibtorchFreeLoader& operator=(AOTILibtorchFreeLoader&& other) = delete;

  AOTILibtorchFreeLoader(
      const std::string& model_package_path,
      const std::string& model_name = "model",
      const bool run_single_threaded = false,
      const size_t num_runners = 1);
  ~AOTILibtorchFreeLoader();

  // boxed_run is not supported yet
  std::vector<aoti::libtorch_free::SlimTensor> run(
      const std::vector<aoti::libtorch_free::SlimTensor>& inputs,
      void* stream_handle = nullptr);

  std::vector<std::string> get_call_spec();
  std::unordered_map<std::string, std::string> get_metadata();

 protected:
  std::string temp_dir_;
  std::unordered_map<std::string, std::string> metadata_;
  std::unique_ptr<AOTLibtorchFreeRunner> runner_;
};

} // namespace aoti::libtorch_free
