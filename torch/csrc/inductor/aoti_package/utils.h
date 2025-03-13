#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <string>
#include <unordered_map>

namespace torch::inductor {
void extrac_zip_file(
    const std::string& model_package_path,
    const std::string& model_name,
    std::string& output_dir,
    std::string& so_path,
    std::string& consts_path,
    std::string& cpp_path,
    std::string& cubin_dir);

void load_metadata(
    const std::string& cpp_path,
    std::unordered_map<std::string, std::string>& metadata);

bool file_exists(const std::string& path);
bool recursive_rmdir(const std::string& path);

#ifdef _WIN32
const std::string k_separator = "\\";
#else
const std::string k_separator = "/";
#endif
} // namespace torch::inductor
#endif
