#include <dlfcn.h>
#include <stdexcept>

#include <torch/csrc/inductor/aoti_libtorch_free/c_shim.h>
#include <torch/csrc/inductor/aoti_libtorch_free/package_loader.h>
#include <torch/csrc/inductor/aoti_libtorch_free/package_loader_utils.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_libtorch_free/utils_cuda.h>
#endif // USE_CUDA

namespace {
std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<aoti::libtorch_free::SlimTensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new aoti::libtorch_free::SlimTensor(std::move(tensor));
    result.push_back(allocated);
  }
  return result;
}

std::vector<aoti::libtorch_free::SlimTensor>
alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length) {
  // Find duplicates by recording the last known index for each handle.
  std::unordered_map<AtenTensorHandle, size_t> lastKnownIdx;
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[handles[i]] = i;
  }

  std::vector<aoti::libtorch_free::SlimTensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    if (handles[i] == nullptr) {
      result.emplace_back();
      continue;
    }

    aoti::libtorch_free::SlimTensor tensor = *handles[i];
    result.emplace_back(std::move(tensor));
    if (lastKnownIdx[handles[i]] == i) {
      aoti_torch_delete_tensor_object(handles[i]);
    }
    handles[i] = nullptr;
  }

  return result;
}
} // namespace

namespace aoti::libtorch_free {
AOTLibtorchFreeRunner::AOTLibtorchFreeRunner(
    const std::string& model_so_path,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded,
    const size_t num_runners)
    : model_so_(dlopen(
          model_so_path.c_str(),
          RTLD_NOW | RTLD_LOCAL
#ifndef __APPLE__
              | RTLD_DEEPBIND  // RTLD_DEEPBIND is required to prioritize C shim
#endif
              )
              )
{
  if (model_so_ == nullptr) {
    const char* error = dlerror();
    // NOLINTNEXTLINE(performance-avoid-endl)
    std::cerr << "Failed to load shared library: " << error << std::endl;
  }

#define LOAD_SYMBOL(var, name_str)                                        \
  var = reinterpret_cast<decltype(var)>(dlsym(model_so_, name_str));      \
  if (!var) {                                                             \
    throw std::runtime_error(std::string("could not dlsym ") + name_str); \
  }

  LOAD_SYMBOL(create_func_, "AOTInductorModelContainerCreateWithDevice");
  LOAD_SYMBOL(delete_func_, "AOTInductorModelContainerDelete");
  LOAD_SYMBOL(get_num_outputs_func_, "AOTInductorModelContainerGetNumOutputs");
  LOAD_SYMBOL(get_call_spec_func_, "AOTInductorModelContainerGetCallSpec");
  const char* run_func_name = run_single_threaded
      ? "AOTInductorModelContainerRunSingleThreaded"
      : "AOTInductorModelContainerRun";
  LOAD_SYMBOL(run_func_, run_func_name);
#undef LOAD_SYMBOL

  AOTI_RUNTIME_ERROR_CODE_CHECK(create_func_(
      &container_handle_,
      num_runners,
      device_str.c_str(),
      cubin_dir.empty() ? nullptr : cubin_dir.c_str()));
}

AOTLibtorchFreeRunner::~AOTLibtorchFreeRunner() {
  if (container_handle_) {
    if (delete_func_(container_handle_) != AOTI_TORCH_SUCCESS) {
      // NOLINTNEXTLINE(performance-avoid-endl)
      std::cerr << "AOTInductorModelContainerDelete failed" << std::endl;
    }
  }
  if (model_so_) {
    if (dlclose(model_so_) != 0) {
      // NOLINTNEXTLINE(performance-avoid-endl)
      std::cerr << "Failed to close shared lib: " << dlerror() << std::endl;
    }
  }
}

std::vector<aoti::libtorch_free::SlimTensor> AOTLibtorchFreeRunner::run(
    const std::vector<aoti::libtorch_free::SlimTensor>& inputs,
    void* stream_handle) {
  std::vector<AtenTensorHandle> input_handles =
      unsafe_alloc_new_handles_from_tensors(inputs);

  size_t num_outputs = 0;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_outputs_func_(container_handle_, &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

#ifdef USE_CUDA
  std::unique_ptr<AOTICudaStream> cuda_stream;
  if (stream_handle == nullptr) {
    cuda_stream = std::make_unique<AOTICudaStream>();
    stream_handle = reinterpret_cast<void*>(cuda_stream->get());
  }
#endif

  AOTI_RUNTIME_ERROR_CODE_CHECK(run_func_(
      container_handle_,
      input_handles.data(),
      input_handles.size(),
      output_handles.data(),
      output_handles.size(),
      reinterpret_cast<AOTInductorStreamHandle>(stream_handle),
      nullptr)); // nullptr for proxy_executor_handle_, which is not supported

  return alloc_tensors_by_stealing_from_handles(
      output_handles.data(), output_handles.size());
}

std::vector<std::string> AOTLibtorchFreeRunner::get_call_spec() {
  const char* in_spec = nullptr;
  const char* out_spec = nullptr;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_call_spec_func_(container_handle_, &in_spec, &out_spec));
  return {in_spec, out_spec};
}

AOTILibtorchFreeLoader::AOTILibtorchFreeLoader(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded,
    const size_t num_runners) {
  if (run_single_threaded) {
    if (num_runners != 1) {
      throw std::runtime_error(
          "num_runners must be 1 when run_single_threaded is true");
    }
  } else {
    if (num_runners < 1) {
      throw std::runtime_error(
          "num_runners must be >=1 when run_single_threaded is false");
    }
  }

  std::string so_path;
  std::string consts_path;
  std::string cpp_path;
  std::string cubin_dir;
  aoti::libtorch_free::extrac_zip_file(
      model_package_path,
      model_name,
      temp_dir_,
      so_path,
      consts_path,
      cpp_path,
      cubin_dir);

  // Load metadata which can be queried by user
  size_t lastindex = cpp_path.find_last_of('.');
  std::string metadata_json_path =
      cpp_path.substr(0, lastindex) + "_metadata.json";
  metadata_ = aoti::libtorch_free::load_metadata(metadata_json_path);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];
  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }

  runner_ = std::make_unique<AOTLibtorchFreeRunner>(
      so_path, device, cubin_dir, run_single_threaded, num_runners);
}

AOTILibtorchFreeLoader::~AOTILibtorchFreeLoader() {
  // Clean up the temporary directory
  if (!temp_dir_.empty()) {
    aoti::libtorch_free::recursive_rmdir(temp_dir_);
  }
}

std::vector<std::string> AOTILibtorchFreeLoader::get_call_spec() {
  return runner_->get_call_spec();
}

std::unordered_map<std::string, std::string> AOTILibtorchFreeLoader::
    get_metadata() {
  return metadata_;
}

std::vector<aoti::libtorch_free::SlimTensor> AOTILibtorchFreeLoader::run(
    const std::vector<aoti::libtorch_free::SlimTensor>& inputs,
    void* stream_handle) {
  return runner_->run(inputs);
}

} // namespace aoti::libtorch_free
