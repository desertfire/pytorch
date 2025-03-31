#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_libtorch_free/utils_cuda.h>
#endif

#include <torch/csrc/inductor/aoti_libtorch_free/device.h>
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
    AOTICudaGuard guard(device_index);
    throw_cuda_error(cudaMalloc(&data, nbytes));
    return data;
  }

  static void free(void* ptr, DeviceIndex device_index) {
    AOTICudaGuard guard(device_index);
    print_cuda_error(cudaFree(ptr));
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

    throw_cuda_error(cudaMemcpy(dst, src, nbytes, direction));
  }
};
#else
template <>
struct DeviceTraits<DeviceType::cuda> {
  static void* allocate(size_t nbytes, DeviceIndex device_index) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }

  static void free(void* ptr, DeviceIndex device_index) {
    std::cerr << "Build with USE_CUDA=1 to enable CUDA support\n";
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
class MaybeOwningStorage {
 public:
  MaybeOwningStorage(
      size_t nbytes,
      DeviceType device_type,
      DeviceIndex device_index)
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

  MaybeOwningStorage(
      void* data,
      DeviceType device_type,
      DeviceIndex device_index)
      : data_(data),
        device_type_(device_type),
        device_index_(device_index),
        owning_(false) {
    // data pointer is not owned by this object
  }

  MaybeOwningStorage() = delete;
  MaybeOwningStorage& operator=(MaybeOwningStorage&& other) = delete;
  MaybeOwningStorage& operator=(const MaybeOwningStorage&) = delete;
  MaybeOwningStorage(MaybeOwningStorage&& other) = delete;
  MaybeOwningStorage(const MaybeOwningStorage&) = delete;

  ~MaybeOwningStorage() {
    if (owning_ && data_ != nullptr) {
      if (device_type_ == DeviceType::cpu) {
        DeviceTraits<DeviceType::cpu>::free(data_, device_index_);
      } else if (device_type_ == DeviceType::cuda) {
        DeviceTraits<DeviceType::cuda>::free(data_, device_index_);
      }
    }
  }

  void clone(
      const NonAtomicSharedPtr<MaybeOwningStorage>& other,
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

  void unsafe_set_to_non_owning() {
    // This is only used when interacting with at::Tensor. When testing
    // libtorch-free AOTI from pytorch, we need to convert the output SlimTensor
    // into at::Tensor, which means the storage ownership should be stolen by
    // at::Tensor. When all the SlimTensors referencing the storage are
    // destroyed, the storage should NOT be freed. It should be freed when the
    // at::Tensor is destroyed.
    owning_ = false;
  }

 private:
  void* data_ = nullptr;
  DeviceType device_type_;
  DeviceIndex device_index_;
  bool owning_;
};

using Storage = NonAtomicSharedPtr<MaybeOwningStorage>;

} // namespace aoti::libtorch_free
