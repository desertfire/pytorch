#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <torch/standalone/slim_tensor/storage.h>
#include <torch/standalone/slim_tensor/utils.h>

namespace torch::standalone {

class SlimTensor {
 public:
  SlimTensor(
      Storage&& storage,
      const ArrayRef& sizes,
      const ArrayRef& strides,
      c10::ScalarType dtype,
      int64_t storage_offset = 0)
      : storage_(std::move(storage)),
        sizes_(sizes),
        strides_(strides),
        dtype_(dtype),
        storage_offset_(storage_offset),
        numel_(compute_numel(sizes_)) {}

  SlimTensor() = delete;
  SlimTensor(const SlimTensor&) = default;
  SlimTensor& operator=(const SlimTensor&) = default;
  SlimTensor(SlimTensor&&) = default;
  SlimTensor& operator=(SlimTensor&&) = default;

  ~SlimTensor() = default;

  void reset() {
    // Decrement the refcount of the storage
    storage_.reset();
  }

  // Accessors
  Storage storage() const {
    return storage_;
  }

  ArrayRef sizes() const {
    return sizes_;
  }

  int64_t size(size_t dim) const {
    return sizes_[dim];
  }

  ArrayRef strides() const {
    return strides_;
  }

  int64_t stride(size_t dim) const {
    return strides_[dim];
  }

  c10::ScalarType dtype() const {
    return dtype_;
  }

  const c10::Device& device() const {
    return storage_->device();
  }

  c10::DeviceType device_type() const {
    return storage_->device_type();
  }

  c10::DeviceIndex device_index() const {
    return storage_->device_index();
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  size_t numel() const {
    return numel_;
  }

  size_t nbytes() const {
    return compute_nbytes(numel_, dtype_);
  }

  size_t dim() const {
    return sizes_.size();
  }

  void* data_ptr() const {
    return storage_->data();
  }

  bool is_contiguous() const {
    return is_contiguous_;
  }

  void set_sizes_and_strides(
      const ArrayRef& sizes,
      const ArrayRef& strides,
      std::optional<int64_t> storage_offset = std::nullopt) {
    const int64_t new_dim = static_cast<int64_t>(sizes.size());
    TORCH_CHECK(
        new_dim == static_cast<int64_t>(strides.size()),
        "dimensionality of sizes (",
        new_dim,
        ") must match dimensionality of strides (",
        strides.size(),
        ")");

    auto* new_sizes = new int64_t[new_dim];
    auto* new_strides = new int64_t[new_dim];
    std::copy(sizes.begin(), sizes.end(), new_sizes);

    // stride calculation logic
    bool overflowed = false;
    if (new_dim > 0) {
      for (int64_t dim = new_dim - 1; dim >= 0; dim--) {
        if (strides[dim] >= 0) {
          new_strides[dim] = strides[dim];
        } else {
          // for negative strides
          if (dim == new_dim - 1) {
            new_strides[dim] = 1;
          } else {
            overflowed |= c10::mul_overflows(
                new_strides[dim + 1],
                std::max<int64_t>(new_sizes[dim + 1], 1),
                &new_strides[dim]);
          }
        }
      }
    }
    TORCH_CHECK(!overflowed, "Stride calculation overflowed");

    sizes_ = ArrayRef(new_sizes, new_dim, /*owning=*/true);
    strides_ = ArrayRef(new_strides, new_dim, /*owning=*/true);

    if (storage_offset.has_value()) {
      storage_offset_ = *storage_offset;
    }

    refresh_numel();
    refresh_contiguous();
  }

  void set_sizes_contiguous(const ArrayRef& sizes) {
    const size_t new_dim = sizes.size();

    auto* new_sizes = new int64_t[new_dim];
    std::copy(sizes.begin(), sizes.end(), new_sizes);

    sizes_ = ArrayRef(new_sizes, new_dim, /*owning=*/true);
    refresh_numel();

    this->empty_tensor_restride(c10::MemoryFormat::Contiguous);
  }

  void empty_tensor_restride(c10::MemoryFormat memory_format) {
    switch (memory_format) {
      case c10::MemoryFormat::Contiguous: {
        const auto current_dim = this->dim();
        auto* new_strides_data = new int64_t[current_dim];

        if (current_dim > 0) {
          const int64_t last_idx = static_cast<int64_t>(current_dim) - 1;
          new_strides_data[last_idx] = 1;
          for (int64_t i = last_idx - 1; i >= 0; i--) {
            new_strides_data[i] = new_strides_data[i + 1] * this->size(i + 1);
          }
        }

        strides_ = ArrayRef(new_strides_data, current_dim, /*owning=*/true);
        break;
      }
      // TODO: implement the other cases.
      case c10::MemoryFormat::ChannelsLast:
      case c10::MemoryFormat::Preserve:
      case c10::MemoryFormat::ChannelsLast3d:
      default:
        TORCH_CHECK(
            memory_format == c10::MemoryFormat::Contiguous,
            "Only support MemoryFormat::Contiguous now");
        break;
    }

    refresh_contiguous();
  }

  SlimTensor as_strided_(
      const ArrayRef& sizes,
      const ArrayRef& strides,
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

  SlimTensor to(const c10::Device& device) const {
    // Does not mutate the current tensor, but returns a new tensor
    if (device == storage_->device()) {
      return *this;
    }
    Storage new_storage(new MaybeOwningStorage(nbytes(), device));
    new_storage->clone(storage_, nbytes(), storage_offset_);
    return SlimTensor(
        std::move(new_storage), sizes_, strides_, dtype_, storage_offset_);
  }

  SlimTensor to(c10::ScalarType dtype) const {
    throw std::runtime_error("TBD: to(dtype)");
  }

<<<<<<< HEAD
  SlimTensor permute(const ArrayRef& dims) const {
    const int64_t ndim = this->dim();
=======
  SlimTensor permute(ArrayRef dims) const {
    const int64_t ndim = static_cast<int64_t>(this->dim());
>>>>>>> 22c1a63f845 (add member functions for slim tensor)

    TORCH_CHECK(
        ndim == static_cast<int64_t>(dims.size()),
        "permute: dims length must be equal to tensor.dim()")

    const auto old_sizes = this->sizes();
    const auto old_strides = this->strides();

    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen_dims(ndim, false);

    for (int64_t i = 0; i < ndim; i++) {
      int64_t d = torch::standalone::maybe_wrap_dim(dims[i], ndim);
      TORCH_CHECK(!seen_dims[d], "permute: duplicate dims are not allowed");
      seen_dims[d] = true;
      new_sizes[i] = old_sizes[d];
      new_strides[i] = old_strides[d];
    }

    SlimTensor result = *this;
    result.as_strided_(
        ArrayRef(new_sizes.data(), ndim, /*owning=*/true),
        ArrayRef(new_strides.data(), ndim, /*owning=*/true),
        this->storage_offset());
<<<<<<< HEAD
=======

>>>>>>> 22c1a63f845 (add member functions for slim tensor)
    return result;
  }

 private:
  // device_type_ and device_index_ are stored in storage_
  Storage storage_;
  // Sizes and strides are either generated by AOTI as static arrays,
  // or dynamically generated, e.g. when in fallback eager ops.
  ArrayRef sizes_;
  ArrayRef strides_;
  c10::ScalarType dtype_;
  int64_t storage_offset_;
  size_t numel_;
  bool is_contiguous_ = true;

  void refresh_numel() {
    numel_ = torch::standalone::compute_numel(sizes_);
  }

  bool compute_is_contiguous() const {
    if (dim() <= 1) {
      return true;
    }

    int64_t expected_stride = 1;
    for (int64_t i = static_cast<int64_t>(dim()) - 1; i >= 0; i--) {
      if (size(i) == 0) {
        return true;
      }
      if (size(i) != 1 && stride(i) != expected_stride) {
        return false;
      }
      expected_stride *= size(i);
    }
    return true;
  }

  void refresh_contiguous() {
    // In SlimTensor, we only care about the single is_contiguous_ flag.
    // (because TensorImpl (aten) implementation has other stuff)
    is_contiguous_ = compute_is_contiguous();
  }
};

// The returned SlimTensor owns the underlying storage
inline SlimTensor create_empty_tensor(
    const ArrayRef& sizes,
    const ArrayRef& strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0,
    bool own_sizes_and_strides = false) {
  ArrayRef new_sizes =
      ArrayRef(sizes.data(), sizes.size(), own_sizes_and_strides);
  ArrayRef new_strides =
      ArrayRef(strides.data(), strides.size(), own_sizes_and_strides);
  size_t nbytes = compute_nbytes(sizes, dtype);
  Storage storage(new MaybeOwningStorage(nbytes, device));
  return SlimTensor(
      std::move(storage), new_sizes, new_strides, dtype, storage_offset);
}

// The returned SlimTensor does not own the underlying storage
inline SlimTensor create_tensor_from_blob(
    void* data,
    const ArrayRef& sizes,
    const ArrayRef& strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  if (data == nullptr) {
    throw std::runtime_error("data pointer can not be nullptr");
  }
  Storage storage(new MaybeOwningStorage(data, device));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}
} // namespace torch::standalone
