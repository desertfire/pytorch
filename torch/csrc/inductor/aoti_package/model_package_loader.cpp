#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <c10/util/error.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_package/utils.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <fmt/format.h>
#include <miniz.h>
#include <nlohmann/json.hpp>
#include <fstream>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// TODO: C++17 has the filesystem header, which may replace these
#ifdef _WIN32
// On Windows, the POSIX implementations are considered deprecated. We simply
// map to the newer variant.
#include <direct.h>
#include <io.h>
#include <process.h>
#define access _access
#define F_OK 0
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace torch::inductor {
namespace {
const nlohmann::json& load_json_file(const std::string& json_path) {
  if (!file_exists(json_path)) {
    throw std::runtime_error("File not found: " + json_path);
  }

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

std::tuple<std::string, std::string> get_cpp_compile_command(
    const std::string& filename,
    const std::vector<std::string>& sources,
    const nlohmann::json& compile_options,
    const std::string& output_dir = "") {
  // Construct the cpp command

  std::string compiler = compile_options["compiler"].get<std::string>();
  bool compile_only = compile_options["compile_only"].get<bool>();

  std::string source_args;
  for (const std::string& source : sources) {
    source_args += source + " ";
  }

  std::string file_ext = compile_only ? ".o" : ".so";
  std::string target_file = output_dir + filename + file_ext;
  std::string target_dir = output_dir;
  if (target_dir.empty()) {
    size_t parent_path_idx = filename.find_last_of(k_separator);
    target_dir = filename.substr(0, parent_path_idx);
  }

  std::string cflags_args;
  for (auto& arg : compile_options["cflags"]) {
    cflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string definitions_args;
  for (auto& arg : compile_options["definitions"]) {
    definitions_args += "-D " + arg.get<std::string>() + " ";
  }

  std::string include_dirs_args;
  for (auto& arg : compile_options["include_dirs"]) {
    include_dirs_args += "-I" + arg.get<std::string>() + " ";
  }

  std::string ldflags_args;
  for (auto& arg : compile_options["ldflags"]) {
    ldflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string libraries_dirs_args;
  for (auto& arg : compile_options["libraries_dirs"]) {
    libraries_dirs_args += "-L" + arg.get<std::string>() + " ";
  }

  std::string libraries_args;
  for (auto& arg : compile_options["libraries"]) {
    libraries_args += "-l" + arg.get<std::string>() + " ";
  }

  std::string passthrough_parameters_args;
  for (auto& arg : compile_options["passthrough_args"]) {
    std::string arg_str = arg.get<std::string>();
    std::string target = "script.ld";
    std::string replacement = target_dir;
    replacement.append(k_separator).append(target);
    size_t pos = arg_str.find(target);
    if (pos != std::string::npos) {
      arg_str.replace(pos, target.length(), replacement);
    }
    passthrough_parameters_args += arg_str + " ";
  }

  std::string compile_only_arg = compile_only ? "-c" : "";

  std::string cmd = fmt::format(
      "{} {} {} {} {} {} {} {} {} {} -o {}",
      compiler,
      source_args,
      definitions_args,
      cflags_args,
      include_dirs_args,
      passthrough_parameters_args,
      ldflags_args,
      libraries_args,
      libraries_dirs_args,
      compile_only_arg,
      target_file);

  return std::make_tuple(cmd, target_file);
}

std::string compile_so(
    const std::string& cpp_path,
    const std::string& consts_path) {
  // Compile the cpp file into a .so

  size_t lastindex = cpp_path.find_last_of('.');
  std::string filename = cpp_path.substr(0, lastindex);

  std::string compile_flags_path = filename + "_compile_flags.json";
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);

  auto [compile_cmd, output_o] =
      get_cpp_compile_command(filename, {cpp_path}, compile_flags);

  std::string linker_flags_path =
      cpp_path.substr(0, lastindex) + "_linker_flags.json";
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);

  auto [link_cmd, output_so] =
      get_cpp_compile_command(filename, {output_o, consts_path}, linker_flags);

  // Run the commands to generate a .so file
  int status = system(compile_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to compile cpp file.");
  }
  status = system(link_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to link files.");
  }

  // Move the mmapped weights onto the .so
  std::string serialized_weights_path = filename + "_serialized_weights.bin";
  if (file_exists(serialized_weights_path)) {
    std::ifstream serialized_weights_file(
        serialized_weights_path, std::ios::binary);
    if (!serialized_weights_file.is_open()) {
      throw std::runtime_error("Failed to open serialized weights file");
    }
    std::vector<char> serialized_weights(
        (std::istreambuf_iterator<char>(serialized_weights_file)),
        std::istreambuf_iterator<char>());
    serialized_weights_file.close();

    std::ofstream output_so_file(output_so, std::ios::binary | std::ios::app);
    if (!output_so_file.is_open()) {
      throw std::runtime_error("Failed to open output .so file");
    }
    // Page align the weights
    std::streampos so_size = output_so_file.tellp();
    std::vector<char> padding(16384 - so_size % 16384, ' ');
    output_so_file.write(
        padding.data(), static_cast<std::streamsize>(padding.size()));
    output_so_file.write(
        serialized_weights.data(),
        static_cast<std::streamsize>(serialized_weights.size()));
    output_so_file.close();
  }

  return output_so;
}
} // namespace

AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded) {
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

  // Compile the .so
  if (so_path.empty()) {
    so_path = compile_so(cpp_path, consts_path);
  }

  // Load metadata which can be queried by user
  load_metadata(cpp_path, metadata_);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];
  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }

  std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      registered_aoti_runner = getAOTIModelRunnerRegistry();
  if (registered_aoti_runner.find(device) == registered_aoti_runner.end()) {
    throw std::runtime_error("Unsupported device found: " + device);
  }

  runner_ = registered_aoti_runner[device](
      so_path, 1, device, cubin_dir, run_single_threaded);
}

AOTIModelPackageLoader::~AOTIModelPackageLoader() {
  // Clean up the temporary directory
  if (!temp_dir_.empty()) {
    recursive_rmdir(temp_dir_);
  }
}

AOTIModelContainerRunner* AOTIModelPackageLoader::get_runner() {
  return runner_.get();
}

std::vector<at::Tensor> AOTIModelPackageLoader::run(
    const std::vector<at::Tensor>& inputs,
    void* stream_handle) {
  return runner_->run(inputs, stream_handle);
}

std::vector<at::Tensor> AOTIModelPackageLoader::boxed_run(
    std::vector<at::Tensor>&& inputs,
    void* stream_handle) {
  return runner_->boxed_run(std::move(inputs), stream_handle);
}

std::unordered_map<std::string, std::string> AOTIModelPackageLoader::
    get_metadata() {
  return metadata_;
}

std::vector<std::string> AOTIModelPackageLoader::get_call_spec() {
  return runner_->get_call_spec();
}

void AOTIModelPackageLoader::load_constants(
    std::unordered_map<std::string, at::Tensor>& constants_map,
    bool use_inactive,
    bool check_full_update) {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::unordered_map<std::string, at::string> fqn_to_constant_name;
  for (const auto& it : constant_name_to_fqn) {
    fqn_to_constant_name.emplace(it.second, it.first);
  }

  std::unordered_map<std::string, at::Tensor> updated_constants_map;
  for (const auto& it : constants_map) {
    if (fqn_to_constant_name.find(it.first) != fqn_to_constant_name.end()) {
      updated_constants_map.emplace(fqn_to_constant_name[it.first], it.second);
    } else {
      throw std::runtime_error("Constant not found: " + it.first);
    }
  }

  return runner_->update_constant_buffer(
      updated_constants_map, use_inactive, check_full_update);
}

std::vector<std::string> AOTIModelPackageLoader::get_constant_fqns() {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::vector<std::string> constant_fqns;
  constant_fqns.reserve(constant_name_to_fqn.size());
  for (const auto& it : constant_name_to_fqn) {
    constant_fqns.push_back(it.second);
  }
  return constant_fqns;
}

} // namespace torch::inductor
#endif
