#include <miniz.h>
#include <torch/csrc/inductor/aoti_neutron/package_loader_utils.h>
#include <fstream>

#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
// On Windows, the POSIX implementations are considered deprecated. We simply
// map to the newer variant.
#include <direct.h>
#include <io.h>
#include <process.h>
#define access _access
#define F_OK 0

#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#endif // _WIN32

namespace torch::native::neutron {
namespace {
std::string create_temp_dir() {
#ifdef _WIN32
  throw std::runtime_error("Not implemented");
  auto tmp_dir = std::filesystem::temp_directory_path() /
      std::filesystem::unique_path("tmp-%%%%-%%%%-%%%%-%%%%");
  std::filesystem::create_directories(tmp_dir);
  return tmp_dir.string();
#else
  std::string temp_dir = "/tmp/XXXXXX";
  if (mkdtemp(temp_dir.data()) == nullptr) {
    throw std::runtime_error(
        std::string("Failed to create temporary directory: ") +
        strerror(errno));
  }
  return temp_dir;
#endif
}

bool recursive_mkdir(const std::string& dir) {
  // Creates directories recursively, copied from jit_utils.cpp
  // Check if current dir exists
  const char* p_dir = dir.c_str();
  const bool dir_exists = (access(p_dir, F_OK) == 0);
  if (dir_exists) {
    return true;
  }

  // Try to create current directory
#ifdef _WIN32
  int ret = _mkdir(dir.c_str());
#else
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  // Success
  if (ret == 0) {
    return true;
  }

  // Find folder separator and check if we are at the top
  auto pos = dir.find_last_of("/\\");
  if (pos == std::string::npos) {
    return false;
  }

  // Try to create parent directory
  if (!(recursive_mkdir(dir.substr(0, pos)))) {
    return false;
  }

  // Try to create complete path again
#ifdef _WIN32
  ret = _mkdir(dir.c_str());
#else
  ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  return ret == 0;
}
} // namespace

bool file_exists(const std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc {};
  return lstat(path.c_str(), &rc) == 0;
#endif
}

bool recursive_rmdir(const std::string& path) {
#ifdef _WIN32
  std::error_code ec;
  return fs::remove_all(path, ec) != static_cast<std::uintmax_t>(-1);
#else
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return false;
  }

  struct dirent* entry = nullptr;
  struct stat statbuf {};
  bool success = true;

  // Iterate through directory entries
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;

    // Skip "." and ".."
    if (name == "." || name == "..") {
      continue;
    }

    std::string full_path = path;
    full_path.append("/").append(name);

    // Get file status
    if (stat(full_path.c_str(), &statbuf) != 0) {
      success = false;
      continue;
    }

    if (S_ISDIR(statbuf.st_mode)) {
      // Recursively delete subdirectory
      if (!recursive_rmdir(full_path)) {
        success = false;
      }
    } else {
      // Delete file
      if (unlink(full_path.c_str()) != 0) {
        success = false;
      }
    }
  }

  closedir(dir);

  // Remove the directory itself
  if (rmdir(path.c_str()) != 0) {
    success = false;
  }

  return success;
#endif
}

void extrac_zip_file(
    const std::string& model_package_path,
    const std::string& model_name,
    std::string& output_dir,
    std::string& so_path,
    std::string& consts_path,
    std::string& cpp_path,
    std::string& cubin_dir) {
  // Extract all files within the zipfile to a temporary directory
  mz_zip_archive zip_archive;
  memset(&zip_archive, 0, sizeof(zip_archive));

  if (!mz_zip_reader_init_file(&zip_archive, model_package_path.c_str(), 0)) {
    throw std::runtime_error(
        std::string("Failed to initialize zip archive: ") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  output_dir = create_temp_dir();
  std::string found_paths; // Saving for bookkeeping
  std::string model_directory =
      "data" + k_separator + "aotinductor" + k_separator + model_name;

  for (uint32_t i = 0; i < zip_archive.m_total_files; i++) {
    uint32_t filename_len =
        mz_zip_reader_get_filename(&zip_archive, i, nullptr, 0);
    if (filename_len == 0) {
      throw std::runtime_error("Failed to read filename");
    }
    char* filename = new char[filename_len + 1];
    if (!mz_zip_reader_get_filename(&zip_archive, i, filename, filename_len)) {
      throw std::runtime_error("Failed to read filename");
    }

    std::string filename_str(filename);
    found_paths += filename_str;
    found_paths += " ";

    // Only compile files in the specified model directory
    if (filename_str.length() >= model_directory.length() &&
        filename_str.substr(0, model_directory.length()) == model_directory) {
      std::string output_path_str = output_dir;
      output_path_str += k_separator;
      output_path_str += filename_str;

      // Create the parent directory if it doesn't exist
      size_t parent_path_idx = output_path_str.find_last_of("/\\");
      if (parent_path_idx == std::string::npos) {
        throw std::runtime_error(
            "Failed to find parent path in " + output_path_str);
      }
      std::string parent_path = output_path_str.substr(0, parent_path_idx);
      if (!recursive_mkdir(parent_path)) {
        throw std::runtime_error("Failed to create directory " + parent_path);
      }

      // Extracts file to the temp directory
      mz_zip_reader_extract_file_to_file(
          &zip_archive, filename, output_path_str.c_str(), 0);

      // Save the file for bookkeeping
      size_t extension_idx = output_path_str.find_last_of('.');
      if (extension_idx != std::string::npos) {
        std::string filename_extension = output_path_str.substr(extension_idx);
        if (filename_extension == ".cpp") {
          cpp_path = output_path_str;
        }
        if (filename_extension == ".o") {
          consts_path = output_path_str;
        }
        if (filename_extension == ".so") {
          so_path = output_path_str;
        }
      }
    }
  }
  cubin_dir = output_dir + k_separator + model_directory;

  // Close the zip archive as we have extracted all files to the temp
  // directory
  if (!mz_zip_reader_end(&zip_archive)) {
    throw std::runtime_error(
        std::string("Failed to close zip archive: {}") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  if (cpp_path.empty() && so_path.empty()) {
    throw std::runtime_error(
        "No AOTInductor generate cpp file or so file found in zip archive. Loaded the following:\n" +
        found_paths);
  }
}

nlohmann::json load_json_file(const std::string& json_path) {
  if (!file_exists(json_path)) {
    throw std::runtime_error("File found: " + json_path);
  }

  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open file: " + json_path);
  }
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

std::unordered_map<std::string, std::string> load_metadata(
    const std::string& path) {
  // Parse metadata json file (if it exists) into the metadata map
  std::unordered_map<std::string, std::string> metadata;
  const nlohmann::json metadata_json_obj = load_json_file(path);
  for (auto& item : metadata_json_obj.items()) {
    metadata[item.key()] = item.value().get<std::string>();
  }
  return metadata;
}
} // namespace torch::native::neutron
