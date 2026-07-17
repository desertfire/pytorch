#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <torch/csrc/jit/backends/aoti/aoti_backend.h>

#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>

namespace torch::jit::aoti {
namespace {

constexpr auto kBackendName = "aoti";
constexpr auto kForward = "forward";
constexpr auto kPackagePath = "package_path";
constexpr auto kModelName = "model_name";
constexpr auto kDeviceIndex = "device_index";

struct PackageSpec {
  std::string package_path;
  std::string model_name;
  c10::DeviceIndex device_index;
};

PackageSpec parsePackageSpec(
    const c10::impl::GenericDict& state,
    const char* state_name) {
  TORCH_CHECK(
      state.size() == 1,
      "AOTI backend ",
      state_name,
      " must contain exactly the \"forward\" package spec");
  const auto item = state.begin();
  TORCH_CHECK(
      item->key().isString() && item->key().toStringRef() == kForward,
      "AOTI backend supports exactly one method named \"forward\"");

  const auto& forward = item->value();
  TORCH_CHECK(
      forward.isGenericDict(),
      "AOTI backend ",
      state_name,
      " \"forward\" package spec must be a Dict[str, Any]");
  const auto package_dict = forward.toGenericDict();
  const std::unordered_set<std::string> expected_keys{
      kPackagePath, kModelName, kDeviceIndex};
  TORCH_CHECK(
      package_dict.size() == expected_keys.size(),
      "AOTI backend ",
      state_name,
      " \"forward\" package spec must contain exactly \"package_path\", \"model_name\", and \"device_index\"");
  for (const auto& item : package_dict) {
    TORCH_CHECK(
        item.key().isString() && expected_keys.count(item.key().toStringRef()),
        "AOTI backend ",
        state_name,
        " \"forward\" package spec contains unsupported key");
  }

  const auto& package_path = package_dict.at(kPackagePath);
  TORCH_CHECK(
      package_path.isString() && !package_path.toStringRef().empty(),
      "AOTI backend ",
      state_name,
      " package_path must be a non-empty string");
  const auto& model_name = package_dict.at(kModelName);
  TORCH_CHECK(
      model_name.isString() && !model_name.toStringRef().empty(),
      "AOTI backend ",
      state_name,
      " model_name must be a non-empty string");
  const auto& device_index = package_dict.at(kDeviceIndex);
  TORCH_CHECK(
      device_index.isInt() && device_index.toInt() >= -1 &&
          device_index.toInt() <= std::numeric_limits<c10::DeviceIndex>::max(),
      "AOTI backend ",
      state_name,
      " device_index must be an integer from -1 through ",
      static_cast<int64_t>(std::numeric_limits<c10::DeviceIndex>::max()));

  return {
      package_path.toStringRef(),
      model_name.toStringRef(),
      static_cast<c10::DeviceIndex>(device_index.toInt())};
}

PackageSpec parseProcessedState(const c10::IValue& processed) {
  TORCH_CHECK(
      processed.isGenericDict(),
      "AOTI backend processed state must be a Dict[str, Any]");
  return parsePackageSpec(processed.toGenericDict(), "processed state");
}

c10::IValue preprocess(
    const Module&,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const BackendDebugHandleGenerator&) {
  parsePackageSpec(method_compile_spec, "method compile spec");
  return method_compile_spec;
}

static auto backend = torch::jit::backend<AOTIBackend>(kBackendName);
static auto preprocessor =
    torch::jit::backend_preprocess_register(kBackendName, preprocess);

} // namespace

AOTIBackend::~AOTIBackend() = default;

bool AOTIBackend::is_available() {
  return true;
}

c10::impl::GenericDict AOTIBackend::compile(
    c10::IValue processed,
    c10::impl::GenericDict method_compile_spec) {
  const auto compile_spec =
      parsePackageSpec(method_compile_spec, "method compile spec");
  const auto package_spec = parseProcessedState(processed);
  TORCH_CHECK(
      package_spec.package_path == compile_spec.package_path &&
          package_spec.model_name == compile_spec.model_name &&
          package_spec.device_index == compile_spec.device_index,
      "AOTI backend processed state must match the method compile spec");
  auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      package_spec.package_path,
      package_spec.model_name,
      /*run_single_threaded=*/true,
      /*num_runners=*/1,
      package_spec.device_index);

  c10::Dict<c10::IValue, c10::IValue> handles(
      c10::StringType::get(), c10::AnyType::get());
  handles.insert(kForward, kForward);
  loader_ = std::move(loader);
  return handles;
}

c10::impl::GenericList AOTIBackend::execute(
    c10::IValue handle,
    c10::impl::GenericList inputs) {
  TORCH_CHECK(
      handle.isString() && handle.toStringRef() == kForward,
      "AOTI backend received an invalid method handle");
  TORCH_CHECK(loader_, "AOTI backend must be compiled before execution");

  std::vector<at::Tensor> tensor_inputs;
  tensor_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor(), "AOTI backend inputs must all be tensors");
    tensor_inputs.push_back(input.toTensor());
  }

  // A null handle tells the CUDA runner to use the caller's current stream.
  auto outputs = loader_->boxed_run(std::move(tensor_inputs), nullptr);
  c10::List<at::Tensor> output_list(std::move(outputs));
  return c10::impl::toList(output_list);
}

} // namespace torch::jit::aoti

#endif
