#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>

#include <torch/csrc/inductor/aoti_libtorch_free/storage.h>
#include <torch/csrc/inductor/aoti_libtorch_free/utils.h>

namespace aoti::libtorch_free {

class SlimTensor {
 public:
  SlimTensor(
      Storage&& storage,
      IntArrayRef sizes,
      IntArrayRef strides,
      ScalarType dtype,
      int64_t storage_offset = 0)
      : storage_(std::move(storage)),
        sizes_(sizes),
        strides_(strides),
        dtype_(dtype),
        storage_offset_(storage_offset) {}

  SlimTensor() = delete;
  SlimTensor(const SlimTensor&) = default;
  SlimTensor& operator=(const SlimTensor&) = default;
  SlimTensor(SlimTensor&&) = default;
  SlimTensor& operator=(SlimTensor&&) = default;

  ~SlimTensor() {}

  void reset() {
    // Decrement the refcount of the storage
    storage_.reset();
  }

  // Accessors
  Storage storage() const {
    return storage_;
  }

  IntArrayRef sizes() const {
    return sizes_;
  }

  int64_t size(size_t dim) const {
    return sizes_[dim];
  }

  IntArrayRef strides() const {
    return strides_;
  }

  int64_t stride(size_t dim) const {
    return strides_[dim];
  }

  ScalarType dtype() const {
    return dtype_;
  }

  DeviceType device_type() const {
    return storage_->device_type();
  }

  DeviceIndex device_index() const {
    return storage_->device_index();
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  size_t numel() const {
    return compute_numel(sizes_);
  }

  size_t nbytes() const {
    return compute_nbytes(sizes_, dtype_);
  }

  size_t dim() const {
    return sizes_.size();
  }

  void* data_ptr() const {
    return storage_->data();
  }

  SlimTensor as_strided_(
      IntArrayRef sizes,
      IntArrayRef strides,
      int64_t storage_offset) {
    sizes_ = sizes;
    strides_ = strides;
    storage_offset_ = storage_offset;
    return *this;
  }

  SlimTensor copy_(const SlimTensor& other) {
    storage_->clone(other.storage(), other.nbytes(), other.storage_offset());
    return *this;
  }

 private:
  // device_type_ and device_index_ are stored in Storage
  Storage storage_;
  // Sizes and strides are expected to be static and generated AOTI
  IntArrayRef sizes_;
  IntArrayRef strides_;
  ScalarType dtype_;
  int64_t storage_offset_;
};

// The returned SlimTensor is owned by the caller
inline SlimTensor create_empty_tensor(
    IntArrayRef sizes,
    IntArrayRef strides,
    ScalarType dtype,
    DeviceType device_type,
    DeviceIndex device_index,
    int64_t storage_offset) {
  size_t nbytes = compute_nbytes(sizes, dtype);
  Storage storage(new StorageBase(nbytes, device_type, device_index));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}

inline SlimTensor create_tensor_from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    ScalarType dtype,
    DeviceType device_type,
    DeviceIndex device_index,
    int64_t storage_offset) {
  if (data == nullptr) {
    throw std::runtime_error("data pointer can not be nullptr");
  }
  Storage storage(new StorageBase(data, device_type, device_index));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}
} // namespace aoti::libtorch_free
