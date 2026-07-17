#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <torch/csrc/jit/backends/aoti/aoti_backend.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/error.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace torch::jit::aoti {
namespace {

constexpr auto kBackendName = "aoti";
constexpr auto kForward = "forward";
constexpr auto kPackagePath = "package_path";
constexpr auto kModelName = "model_name";
constexpr auto kDeviceIndex = "device_index";
constexpr auto kPackageBytes = "package_bytes";

struct PackageSpec {
  std::string package_path;
  std::string model_name;
  c10::DeviceIndex device_index;
};

struct ProcessedState {
  PackageSpec package_spec;
  at::Tensor package_bytes;
};

c10::impl::GenericDict parseForwardState(
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
  return forward.toGenericDict();
}

void validatePackageKeys(
    const c10::impl::GenericDict& package_dict,
    const char* state_name,
    bool expect_package_bytes) {
  std::unordered_set<std::string> expected_keys{
      kPackagePath, kModelName, kDeviceIndex};
  if (expect_package_bytes) {
    expected_keys.insert(kPackageBytes);
  }
  TORCH_CHECK(
      package_dict.size() == expected_keys.size(),
      "AOTI backend ",
      state_name,
      " \"forward\" package spec must contain exactly \"package_path\", \"model_name\", and \"device_index\"",
      expect_package_bytes ? ", plus \"package_bytes\"" : "");
  for (const auto& item : package_dict) {
    TORCH_CHECK(
        item.key().isString() && expected_keys.count(item.key().toStringRef()),
        "AOTI backend ",
        state_name,
        " \"forward\" package spec contains unsupported key");
  }
}

PackageSpec parsePackageMetadata(
    const c10::impl::GenericDict& package_dict,
    const char* state_name) {
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

PackageSpec parsePackageSpec(
    const c10::impl::GenericDict& state,
    const char* state_name) {
  const auto package_dict = parseForwardState(state, state_name);
  validatePackageKeys(package_dict, state_name, false);
  return parsePackageMetadata(package_dict, state_name);
}

ProcessedState parseProcessedState(const c10::IValue& processed) {
  TORCH_CHECK(
      processed.isGenericDict(),
      "AOTI backend processed state must be a Dict[str, Any]");
  const auto package_dict =
      parseForwardState(processed.toGenericDict(), "processed state");
  validatePackageKeys(package_dict, "processed state", true);
  const auto& package_bytes_value = package_dict.at(kPackageBytes);
  TORCH_CHECK(
      package_bytes_value.isTensor(),
      "AOTI backend processed state package_bytes must be a Tensor");
  const auto package_bytes = package_bytes_value.toTensor();
  TORCH_CHECK(
      package_bytes.defined(),
      "AOTI backend processed state package_bytes must be defined");
  TORCH_CHECK(
      package_bytes.device().is_cpu(),
      "AOTI backend processed state package_bytes must be on CPU");
  TORCH_CHECK(
      package_bytes.scalar_type() == at::kByte,
      "AOTI backend processed state package_bytes must have dtype uint8");
  TORCH_CHECK(
      package_bytes.dim() == 1,
      "AOTI backend processed state package_bytes must be 1-D");
  TORCH_CHECK(
      package_bytes.is_contiguous(),
      "AOTI backend processed state package_bytes must be contiguous");
  TORCH_CHECK(
      package_bytes.numel() > 0,
      "AOTI backend processed state package_bytes must be nonempty");
  return {parsePackageMetadata(package_dict, "processed state"), package_bytes};
}

at::Tensor readPackageBytes(const std::string& package_path) {
  std::ifstream package_file(package_path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(
      package_file.is_open(),
      "AOTI backend failed to open package for preprocessing: ",
      package_path);

  const std::streamoff package_size = package_file.tellg();
  TORCH_CHECK(
      package_size >= 0,
      "AOTI backend failed to determine package size during preprocessing: ",
      package_path);
  TORCH_CHECK(
      package_size > 0,
      "AOTI backend package must be nonempty during preprocessing: ",
      package_path);
  const auto package_size_unsigned = static_cast<std::uintmax_t>(package_size);
  TORCH_CHECK(
      package_size_unsigned <= static_cast<std::uintmax_t>(
                                   std::numeric_limits<int64_t>::max()) &&
          package_size_unsigned <=
              static_cast<std::uintmax_t>(
                  std::numeric_limits<std::streamsize>::max()),
      "AOTI backend package is too large to embed during preprocessing: ",
      package_path);

  const auto package_size_int64 = static_cast<int64_t>(package_size);
  auto package_bytes = at::empty(
      {package_size_int64},
      at::TensorOptions().device(at::kCPU).dtype(at::kByte));
  package_file.seekg(0, std::ios::beg);
  TORCH_CHECK(
      package_file.good(),
      "AOTI backend failed to seek package during preprocessing: ",
      package_path);
  package_file.read(
      reinterpret_cast<char*>(package_bytes.data_ptr<uint8_t>()),
      static_cast<std::streamsize>(package_size));
  TORCH_CHECK(
      package_file.gcount() == static_cast<std::streamsize>(package_size),
      "AOTI backend failed to read package during preprocessing: ",
      package_path);
  return package_bytes;
}

c10::TempFile materializePackage(const at::Tensor& package_bytes) {
  auto package_file = c10::try_make_tempfile("torch-aoti-package-");
#if defined(_WIN32)
  TORCH_CHECK(
      package_file, "AOTI backend failed to create a temporary package file");
  TORCH_CHECK(
      package_file->open(),
      "AOTI backend failed to open temporary package file for binary writing: ",
      package_file->name,
      ": ",
      c10::utils::str_error(errno));
#else
  TORCH_CHECK(
      package_file,
      "AOTI backend failed to create a temporary package file: ",
      c10::utils::str_error(errno));
#endif

  const auto* data = package_bytes.const_data_ptr<uint8_t>();
  int64_t offset = 0;
  while (offset < package_bytes.numel()) {
    const auto remaining = package_bytes.numel() - offset;
    const auto chunk_size = static_cast<unsigned int>(
        std::min<int64_t>(remaining, std::numeric_limits<int>::max()));
#if defined(_WIN32)
    int written;
    do {
      written = _write(package_file->fd, data + offset, chunk_size);
    } while (written < 0 && errno == EINTR);
#else
    ssize_t written;
    do {
      written = ::write(package_file->fd, data + offset, chunk_size);
    } while (written < 0 && errno == EINTR);
#endif
    const auto write_error = errno;
    TORCH_CHECK(
        written >= 0,
        "AOTI backend failed to write embedded package bytes to temporary file ",
        package_file->name,
        ": ",
        c10::utils::str_error(write_error));
    TORCH_CHECK(
        written > 0,
        "AOTI backend made no progress writing embedded package bytes to temporary file ",
        package_file->name);
    offset += written;
  }
  return std::move(*package_file);
}

c10::IValue preprocess(
    const Module&,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const BackendDebugHandleGenerator&) {
  const auto package_spec =
      parsePackageSpec(method_compile_spec, "method compile spec");
  c10::Dict<c10::IValue, c10::IValue> processed_package(
      c10::StringType::get(), c10::AnyType::get());
  processed_package.insert(kPackagePath, package_spec.package_path);
  processed_package.insert(kModelName, package_spec.model_name);
  processed_package.insert(
      kDeviceIndex, static_cast<int64_t>(package_spec.device_index));
  processed_package.insert(
      kPackageBytes, readPackageBytes(package_spec.package_path));

  c10::Dict<c10::IValue, c10::IValue> processed(
      c10::StringType::get(), c10::AnyType::get());
  processed.insert(kForward, processed_package);
  return processed;
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
  const auto processed_state = parseProcessedState(processed);
  const auto& package_spec = processed_state.package_spec;
  TORCH_CHECK(
      package_spec.package_path == compile_spec.package_path &&
          package_spec.model_name == compile_spec.model_name &&
          package_spec.device_index == compile_spec.device_index,
      "AOTI backend processed state must match the method compile spec");
  auto package_file = materializePackage(processed_state.package_bytes);
  auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      package_file.name,
      package_spec.model_name,
      /*run_single_threaded=*/true,
      /*num_runners=*/1,
      package_spec.device_index);

  c10::Dict<c10::IValue, c10::IValue> handles(
      c10::StringType::get(), c10::AnyType::get());
  handles.insert(kForward, kForward);
  loader_.reset();
  package_file_.reset();
  package_file_.emplace(std::move(package_file));
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
