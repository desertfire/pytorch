#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <dlfcn.h>
#include <torch/csrc/inductor/aoti_libtorch_free/c_shim.h>
#include <torch/csrc/inductor/aoti_libtorch_free/package_loader_utils.h>
#include <torch/csrc/inductor/aoti_package/libtorch_free_package_loader.h>

namespace {

std::vector<aoti::libtorch_free::SlimTensor> convert_aten_tensor_to_slim_tensor(
    const std::vector<at::Tensor>& tensors) {
  std::vector<aoti::libtorch_free::SlimTensor> result;
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
    result.push_back(aoti::libtorch_free::create_tensor_from_blob(
        tensor.data_ptr(),
        sizes,
        strides,
        // dtype is 1-to-1 mapping for now. No guanrantee it will always be
        // the case
        static_cast<aoti::libtorch_free::ScalarType>(
            c10::typeMetaToScalarType(tensor.dtype())),
        // device_type is 1-to-1 mapping for now. No guanrantee it will
        // always be the case
        static_cast<aoti::libtorch_free::DeviceType>(tensor.device().type()),
        static_cast<aoti::libtorch_free::DeviceIndex>(tensor.device().index()),
        tensor.storage_offset()));
  }
  return result;
}

std::vector<at::Tensor> convert_slim_tensor_to_aten_tensor(
    const std::vector<aoti::libtorch_free::SlimTensor>& tensors) {
  std::vector<at::Tensor> result;
  result.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    // NOTE: Output tensor ownership
    // Again, we need to take care of the ownership of device storage and
    // sizes_and_strides. For quick prototyping, we create an aten tensor using
    // from_blob and then clone it to another aten tensor, which will manages
    // the new output tensor lifttime separately from the original SlimTensor.
    c10::IntArrayRef sizes(tensor.sizes().data(), tensor.sizes().size());
    c10::IntArrayRef strides(tensor.strides().data(), tensor.strides().size());
    c10::Device device = c10::Device(
        static_cast<c10::DeviceType>(tensor.device_type()),
        static_cast<c10::DeviceIndex>(tensor.device_index()));
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(tensor.dtype()));
    at::Tensor tmp_tensor = at::for_blob(tensor.data_ptr(), sizes)
                                .strides(strides)
                                .storage_offset(tensor.storage_offset())
                                .options(options)
                                .make_tensor();
    result.push_back(tmp_tensor.clone());
  }
  return result;
}

} // namespace

namespace torch::inductor {
AOTILibtorchFreeLoaderFromAten::AOTILibtorchFreeLoaderFromAten(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded,
    const size_t num_runners)
    : aoti::libtorch_free::AOTILibtorchFreeLoader(
          model_package_path,
          model_name,
          run_single_threaded,
          num_runners) {}

std::vector<at::Tensor> AOTILibtorchFreeLoaderFromAten::run(
    const std::vector<at::Tensor>& inputs,
    void* stream_handle) {
  std::vector<aoti::libtorch_free::SlimTensor> tmp_inputs =
      convert_aten_tensor_to_slim_tensor(inputs);
  std::vector<aoti::libtorch_free::SlimTensor> tmp_outputs =
      runner_->run(tmp_inputs, stream_handle);
  return convert_slim_tensor_to_aten_tensor(tmp_outputs);
}

} // namespace torch::inductor
#endif
