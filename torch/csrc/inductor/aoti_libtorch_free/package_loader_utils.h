#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

// Common util funtions for operations on generated model package, no mater if
// the package file is from the stardard AOTI or the libtorch-free AOTI
namespace aoti::libtorch_free {
#ifdef _WIN32
const std::string k_separator = "\\";
#else
const std::string k_separator = "/";
#endif

bool file_exists(const std::string& path);

bool recursive_rmdir(const std::string& path);

void extrac_zip_file(
    const std::string& model_package_path,
    const std::string& model_name,
    std::string& output_dir,
    std::string& so_path,
    std::string& consts_path,
    std::string& cpp_path,
    std::string& cubin_dir);

nlohmann::json load_json_file(const std::string& json_path);

std::unordered_map<std::string, std::string> load_metadata(
    const std::string& path);
} // namespace aoti::libtorch_free
