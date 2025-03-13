#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <dlfcn.h>
#include <torch/csrc/inductor/aoti_libtorch_free/c_shim.h>
#include <torch/csrc/inductor/aoti_package/libtorch_free_loader.h>
#include <torch/csrc/inductor/aoti_package/utils.h>

namespace {

std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // NOTE: Input tensor ownership
    // There two ownerships to be sorted out: device storage and
    // sizes_and_strides. For quick prototyping, we just keep input aten tensors
    // always alive by not doing any decref on them.
    torch::aot_inductor::MiniArrayRef sizes(
        tensor.sizes().data(), tensor.sizes().size());
    torch::aot_inductor::MiniArrayRef strides(
        tensor.strides().data(), tensor.strides().size());
    aoti::libtorch_free::SlimTensor tmp =
        aoti::libtorch_free::create_tensor_from_blob(
            tensor.data_ptr(),
            sizes,
            strides,
            // dtype is 1-to-1 mapping for now. No guanrantee it will always be
            // the case
            static_cast<aoti::libtorch_free::ScalarType>(
                c10::typeMetaToScalarType(tensor.dtype())),
            // device_type is 1-to-1 mapping for now. No guanrantee it will
            // always be the case
            static_cast<aoti::libtorch_free::DeviceType>(
                tensor.device().type()),
            static_cast<aoti::libtorch_free::DeviceIndex>(
                tensor.device().index()),
            tensor.storage_offset());
    auto allocated = new aoti::libtorch_free::SlimTensor(tmp);
    result.push_back(static_cast<AtenTensorHandle>(allocated));
  }
  return result;
}

std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length) {
  // Find duplicates by recording the last known index for each handle.
  std::unordered_map<AtenTensorHandle, size_t> lastKnownIdx;
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[handles[i]] = i;
  }

  std::vector<at::Tensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    if (handles[i] == nullptr) {
      result.emplace_back();
      continue;
    }

    // NOTE: Output tensor ownership
    // Again, we need to take care of the ownership of device storage and
    // sizes_and_strides. For quick prototyping, we create an aten tensor using
    // from_blob and then clone it to another aten tensor, which will manages
    // the new output tensor lifttime separately from the original SlimTensor.
    const AtenTensorHandle& handle = handles[i];
    c10::IntArrayRef sizes(handle->sizes().data(), handle->sizes().size());
    c10::IntArrayRef strides(
        handle->strides().data(), handle->strides().size());
    c10::Device device = c10::Device(
        static_cast<c10::DeviceType>(handle->device_type()),
        static_cast<c10::DeviceIndex>(handle->device_index()));
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(handle->dtype()));
    at::Tensor tmp_tensor = at::for_blob(handle->data_ptr(), sizes)
                                .strides(strides)
                                .storage_offset(handle->storage_offset())
                                .options(options)
                                .make_tensor();
    result.emplace_back(tmp_tensor.clone());
    if (lastKnownIdx[handles[i]] == i) {
      delete handles[i];
    }
    handles[i] = nullptr;
  }

  return result;
}

} // namespace

namespace torch::inductor {
AOTLibtorchFreeRunner::AOTLibtorchFreeRunner(
    const std::string& model_so_path,
    const std::string& device_str)
    : model_so_(dlopen(
          model_so_path.c_str(),
          RTLD_NOW | RTLD_LOCAL |
              RTLD_DEEPBIND)) // RTLD_DEEPBIND is required to prioritize C shim
                              // symbols defined in model.so over libtorch.so
{
  // TODO: switch to use APIs without going through AOTInductorModelContainer
#define LOAD_SYMBOL(var, name_str)                                   \
  var = reinterpret_cast<decltype(var)>(dlsym(model_so_, name_str)); \
  TORCH_CHECK(var, "could not dlsym " name_str);

  LOAD_SYMBOL(create_func_, "AOTInductorModelContainerCreateWithDevice");
  LOAD_SYMBOL(delete_func_, "AOTInductorModelContainerDelete");
  LOAD_SYMBOL(get_num_outputs_func_, "AOTInductorModelContainerGetNumOutputs");
  LOAD_SYMBOL(run_func_, "AOTInductorModelContainerRun");
  LOAD_SYMBOL(get_call_spec_func_, "AOTInductorModelContainerGetCallSpec");
#undef LOAD_SYMBOL

  AOTI_RUNTIME_ERROR_CODE_CHECK(
      create_func_(&container_handle_, 1, device_str.c_str(), nullptr));
}

AOTLibtorchFreeRunner::~AOTLibtorchFreeRunner() {
  if (container_handle_) {
    AOTIRuntimeError result = delete_func_(container_handle_);
    TORCH_CHECK(
        result == AOTI_RUNTIME_SUCCESS,
        "AOTInductorModelContainerDelete failed");
  }
  if (model_so_) {
    int result = dlclose(model_so_);
    TORCH_CHECK(result == 0, "Failed to close shared lib:", dlerror());
  }
}

std::vector<at::Tensor> AOTLibtorchFreeRunner::run(
    const std::vector<at::Tensor>& inputs) {
  std::vector<AtenTensorHandle> input_handles =
      unsafe_alloc_new_handles_from_tensors(inputs);

  size_t num_outputs = 0;
  AOTI_RUNTIME_ERROR_CODE_CHECK(
      get_num_outputs_func_(container_handle_, &num_outputs));
  std::vector<AtenTensorHandle> output_handles(num_outputs);

  AOTI_RUNTIME_ERROR_CODE_CHECK(run_func_(
      container_handle_,
      input_handles.data(),
      input_handles.size(),
      output_handles.data(),
      output_handles.size(),
      nullptr,
      nullptr));

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
    const std::string& model_name) {
  std::string so_path;
  std::string consts_path;
  std::string cpp_path;
  std::string cubin_dir;
  extrac_zip_file(
      model_package_path,
      model_name,
      temp_dir_,
      so_path,
      consts_path,
      cpp_path,
      cubin_dir);

  // Load metadata which can be queried by user
  load_metadata(cpp_path, metadata_);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];
  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }

  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }
  runner_ = std::make_unique<AOTLibtorchFreeRunner>(so_path, device);

  /*
    std::unordered_map<std::string, CreateAOTLibtorchFreeRunnerFunc>
        registered_aoti_runner = getAOTLibtorchFreeRunnerRegistry();
    if (registered_aoti_runner.find(device) == registered_aoti_runner.end()) {
      throw std::runtime_error("Unsupported device found: " + device);
    }
    runner_ = registered_aoti_runner[device](so_path, 1, device, cubin_dir);
    */
}

AOTILibtorchFreeLoader::~AOTILibtorchFreeLoader() {
  // Clean up the temporary directory
  if (!temp_dir_.empty()) {
    recursive_rmdir(temp_dir_);
  }
}

std::vector<std::string> AOTILibtorchFreeLoader::get_call_spec() {
  return runner_->get_call_spec();
}

std::vector<at::Tensor> AOTILibtorchFreeLoader::run(
    const std::vector<at::Tensor>& inputs) {
  return runner_->run(inputs);
}

/*
std::unordered_map<std::string, CreateAOTLibtorchFreeRunnerFunc>&
getAOTLibtorchFreeRunnerRegistry() {
  static std::unordered_map<std::string, CreateAOTLibtorchFreeRunnerFunc>
      aoti_model_runner_registry_;
  return aoti_model_runner_registry_;
}*/

} // namespace torch::inductor
#endif
