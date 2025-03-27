#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <torch/csrc/inductor/aoti_libtorch_free/device_type.h>
#include <torch/csrc/inductor/aoti_libtorch_free/non_atomic_shared_ptr.h>

namespace aoti::libtorch_free {

// Device traits template for device-specific operations
template <DeviceType D>
struct DeviceTraits;

// CPU specialization
template <>
struct DeviceTraits<DeviceType::cpu> {
  static void* allocate(size_t nbytes, DeviceIndex /*device_index*/) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    return malloc(nbytes);
  }

  static void free(void* ptr, DeviceIndex /*device_index*/) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    ::free(ptr);
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex /*dst_idx*/,
      DeviceIndex /*src_idx*/) {
    std::memcpy(dst, src, nbytes);
  }
};

// CUDA specialization (conditionally compiled)
#ifdef USE_CUDA
template <>
struct DeviceTraits<DeviceType::cuda> {
  static void* allocate(size_t nbytes, DeviceIndex device_index) {
    void* data = nullptr;
    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice failed");
    }
    err = cudaMalloc(&data, nbytes);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
    return data;
  }

  static void free(void* ptr, DeviceIndex device_index) {
    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
      std::cerr << "cudaSetDevice failed\n";
    }
    err = cudaFree(ptr);
    if (err != cudaSuccess) {
      std::cerr << "cudaFree failed\n";
    }
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex dst_idx,
      DeviceIndex src_idx) {
    // Determine the direction
    cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
    if (src_idx == CPU_DEVICE_INDEX) {
      direction = cudaMemcpyHostToDevice;
    } else if (dst_idx == CPU_DEVICE_INDEX) {
      direction = cudaMemcpyDeviceToHost;
    } else {
      if (src_idx != dst_idx) {
        throw std::runtime_error(
            "CUDA memcpy failed: src_device_index != dst_device_index");
      }
    }

    cudaError_t err = cudaMemcpy(dst, src, nbytes, direction);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
  }
};
#else
template <>
struct DeviceTraits<DeviceType::cuda> {
  static void* allocate(size_t nbytes, DeviceIndex device_index) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }

  static void free(void* ptr, DeviceIndex device_index) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      DeviceIndex dst_idx,
      DeviceIndex src_idx) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }
};
#endif

// Storage can be either owning or non-owning. For AOTI-generated intermediate
// tensors, the storage is always owning. For constant tensors, the storage is
// non-owning.
class StorageBase {
 public:
  StorageBase(size_t nbytes, DeviceType device_type, DeviceIndex device_index)
      : device_type_(device_type), device_index_(device_index), owning_(true) {
    // Allocating memory here so owning_ has to be true.
    if (device_type == DeviceType::cpu) {
      data_ = DeviceTraits<DeviceType::cpu>::allocate(nbytes, device_index);
    } else if (device_type == DeviceType::cuda) {
      data_ = DeviceTraits<DeviceType::cuda>::allocate(nbytes, device_index);
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
        DeviceTraits<DeviceType::cpu>::free(data_, device_index_);
      } else if (device_type_ == DeviceType::cuda) {
        DeviceTraits<DeviceType::cuda>::free(data_, device_index_);
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

    void* src_ptr = static_cast<char*>(other->data_) + storage_offset;

    if (device_type_ == DeviceType::cpu &&
        other->device_type_ == DeviceType::cpu) {
      // CPU to CPU copy
      DeviceTraits<DeviceType::cpu>::memcpy(
          data_, src_ptr, nbytes, device_index_, other->device_index_);
    } else {
      // At least one of the devices is CUDA
      DeviceTraits<DeviceType::cuda>::memcpy(
          data_, src_ptr, nbytes, device_index_, other->device_index_);
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
  void* data_ = nullptr;
  DeviceType device_type_;
  DeviceIndex device_index_;
  bool owning_;
};

using Storage = NonAtomicSharedPtr<StorageBase>;

} // namespace aoti::libtorch_free