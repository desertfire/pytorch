#pragma once
#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <torch/csrc/inductor/aoti_libtorch_free/device_type.h>
#include <torch/csrc/inductor/aoti_libtorch_free/non_atomic_shared_ptr.h>
#include <torch/csrc/inductor/aoti_libtorch_free/package_loader_utils.h>

namespace aoti::libtorch_free {

// Storage can be either owning or non-owning. For AOTI-generated intermediate
// tensors, the storage is always owning. For constant tensors, the storage is
// non-owning.
class StorageBase {
 public:
  StorageBase(size_t nbytes, DeviceType device_type, DeviceIndex device_index)
      : device_type_(device_type), device_index_(device_index), owning_(true) {
    // Allocating memory here so owning_ has to be true.
    if (device_type == DeviceType::cpu) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      data_ = malloc(nbytes);
    } else if (device_type == DeviceType::cuda) {
#ifdef USE_CUDA
      cudaError_t err = cudaSetDevice(device_index);
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed");
      }
      err = cudaMalloc(&data_, nbytes);
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
      }
#else
      throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
#endif
    } else {
      throw std::runtime_error("Unsupported device type");
    }
  }

  StorageBase(void* data, DeviceType device_type, DeviceIndex device_index)
      : data_(data),
        device_type_(device_type),
        device_index_(device_index),
        owning_(false) {
    // data pointer is not owned by this object
  }

  StorageBase() = delete;
  StorageBase& operator=(StorageBase&& other) = delete;
  StorageBase& operator=(const StorageBase&) = delete;
  StorageBase(StorageBase&& other) = delete;
  StorageBase(const StorageBase&) = delete;

  ~StorageBase() {
    if (owning_ && data_ != nullptr) {
      if (device_type_ == DeviceType::cpu) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(data_);
      } else if (device_type_ == DeviceType::cuda) {
#ifdef USE_CUDA
        cudaError_t err = cudaSetDevice(device_index_);
        if (err != cudaSuccess) {
          std::cerr << "cudaSetDevice failed\n";
        }
        err = cudaFree(data_);
        if (err != cudaSuccess) {
          std::cerr << "cudaFree failed\n";
        }
#endif
      }
    }
  }

  void clone(
      const NonAtomicSharedPtr<StorageBase>& other,
      size_t nbytes,
      int64_t storage_offset) {
    if (data_ == nullptr || other->data_ == nullptr) {
      throw std::runtime_error(
          "Storage clone failed: data_ can not be nullptr");
    }

    if (device_type_ == DeviceType::cuda ||
        other->device_type_ == DeviceType::cuda) {
      cuda_clone(other, nbytes, storage_offset);
    } else if (device_type_ == DeviceType::cpu) {
      data_ = std::memcpy(
          data_, static_cast<char*>(other->data_) + storage_offset, nbytes);
    } else {
      throw std::runtime_error("Unsupported device type");
    }
  }

  void* data() const {
    return data_;
  }

  DeviceType device_type() const {
    return device_type_;
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

 private:
  void cuda_clone(
      const NonAtomicSharedPtr<StorageBase>& other,
      size_t nbytes,
      int64_t storage_offset) {
#ifdef USE_CUDA
    DeviceType source_device_type = other->device_type_;
    DeviceIndex source_device_index = other->device_index_;
    DeviceType target_device_type = device_type_;
    DeviceIndex target_device_index = device_index_;

    cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
    if (source_device_type == DeviceType::cpu) {
      direction = cudaMemcpyHostToDevice;
    } else if (target_device_type == DeviceType::cpu) {
      direction = cudaMemcpyDeviceToHost;
    }

    cudaError_t err = cudaMemcpy(
        data_,
        static_cast<char*>(other->data_) + storage_offset,
        nbytes,
        direction);
    // Check for errors
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }
#else
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
#endif
  }

  void* data_ = nullptr;
  DeviceType device_type_;
  DeviceIndex device_index_;
  bool owning_;
};

using Storage = NonAtomicSharedPtr<StorageBase>;

} // namespace aoti::libtorch_free
