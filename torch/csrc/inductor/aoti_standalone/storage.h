#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/utils.h>
#endif

#include <torch/csrc/inductor/aoti_standalone/shared_ptr.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>

namespace torch::native::standalone {

const c10::Device CPU_DEVICE = c10::Device(c10::DeviceType::CPU, 0);

// Device traits template for device-specific operations
template <c10::DeviceType D>
struct DeviceTraits;

// CPU specialization
template <>
struct DeviceTraits<c10::DeviceType::CPU> {
  static void* allocate(size_t nbytes, const c10::Device& device) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    return malloc(nbytes);
  }

  static void free(void* ptr, const c10::Device& device) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    ::free(ptr);
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    std::memcpy(dst, src, nbytes);
  }
};

// CUDA specialization (conditionally compiled)
#ifdef USE_CUDA
template <>
struct DeviceTraits<c10::DeviceType::CUDA> {
  static void* allocate(size_t nbytes, const c10::Device& device) {
    void* data = nullptr;
    AOTICudaGuard guard(device.index());
    throw_cuda_error(cudaMalloc(&data, nbytes));
    return data;
  }

  static void free(void* ptr, const c10::Device& device) {
    AOTICudaGuard guard(device.index());
    print_cuda_error(cudaFree(ptr));
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    // Determine the direction
    cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
    if (src_device.is_cpu()) {
      direction = cudaMemcpyHostToDevice;
    } else if (dst_device.is_cpu()) {
      direction = cudaMemcpyDeviceToHost;
    } else {
      if (src_device.index() != dst_device.index()) {
        throw std::runtime_error(
            "CUDA memcpy failed: src_device_index != dst_device_index");
      }
    }

    throw_cuda_error(cudaMemcpy(dst, src, nbytes, direction));
  }
};
#else
template <>
struct DeviceTraits<c10::DeviceType::CUDA> {
  static void* allocate(size_t nbytes, const c10::Device& device) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }

  static void free(void* ptr, const c10::Device& device) {
    std::cerr << "Build with USE_CUDA=1 to enable CUDA support\n";
  }

  static void memcpy(
      void* dst,
      const void* src,
      size_t nbytes,
      const c10::Device& dst_device,
      const c10::Device& src_device) {
    throw std::runtime_error("Build with USE_CUDA=1 to enable CUDA support");
  }
};
#endif

// Storage can be either owning or non-owning. For AOTI-generated intermediate
// tensors, the storage is always owning. For constant tensors, the storage is
// non-owning.
class MaybeOwningStorage {
 public:
  MaybeOwningStorage(size_t nbytes, const c10::Device& device)
      : device_(device), owning_(true) {
    // Allocating memory here so owning_ has to be true.
    if (device.is_cpu()) {
      data_ = DeviceTraits<c10::DeviceType::CPU>::allocate(nbytes, device);
    } else if (device.is_cuda()) {
      data_ = DeviceTraits<c10::DeviceType::CUDA>::allocate(nbytes, device);
    } else {
      throw std::runtime_error("Unsupported device type");
    }
  }

  MaybeOwningStorage(void* data, const c10::Device& device)
      : data_(data), device_(device), owning_(false) {
    // data pointer is not owned by this object
  }

  MaybeOwningStorage() = delete;
  MaybeOwningStorage& operator=(MaybeOwningStorage&& other) = delete;
  MaybeOwningStorage& operator=(const MaybeOwningStorage&) = delete;
  MaybeOwningStorage(MaybeOwningStorage&& other) = delete;
  MaybeOwningStorage(const MaybeOwningStorage&) = delete;

  ~MaybeOwningStorage() {
    if (owning_ && data_ != nullptr) {
      if (device_.is_cpu()) {
        DeviceTraits<c10::DeviceType::CPU>::free(data_, device_);
      } else if (device_.is_cuda()) {
        DeviceTraits<c10::DeviceType::CUDA>::free(data_, device_);
      }
    }
  }

  void clone(
      const SharedPtr<MaybeOwningStorage>& other,
      size_t nbytes,
      int64_t storage_offset) {
    if (data_ == nullptr || other->data_ == nullptr) {
      throw std::runtime_error(
          "Storage clone failed: data_ can not be nullptr");
    }

    void* src_ptr = static_cast<char*>(other->data_) + storage_offset;

    if (device_.is_cpu() && other->device_.is_cpu()) {
      // CPU to CPU copy
      DeviceTraits<c10::DeviceType::CPU>::memcpy(
          data_, src_ptr, nbytes, device_, other->device_);
    } else {
      // At least one of the devices is CUDA
      DeviceTraits<c10::DeviceType::CUDA>::memcpy(
          data_, src_ptr, nbytes, device_, other->device_);
    }
  }

  void* data() const {
    return data_;
  }

  const c10::Device& device() const {
    return device_;
  }

  c10::DeviceType device_type() const {
    return device_.type();
  }

  c10::DeviceIndex device_index() const {
    return device_.index();
  }

  void unsafe_set_to_non_owning() {
    // This is only used when interacting with at::Tensor. When testing
    // standalone AOTI from pytorch, we need to convert the output SlimTensor
    // into at::Tensor, which means the storage ownership should be stolen by
    // at::Tensor. When all the SlimTensors referencing the storage are
    // destroyed, the storage should NOT be freed. It should be freed when the
    // at::Tensor is destroyed.
    owning_ = false;
  }

 private:
  void* data_ = nullptr;
  c10::Device device_;
  bool owning_;
};

using Storage = SharedPtr<MaybeOwningStorage>;

} // namespace torch::native::standalone
