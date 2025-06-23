#pragma once
#include <torch/csrc/inductor/aoti_standalone/c/shim.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <torch/standalone/slim_tensor/utils.h>
namespace torch::standalone {

template <typename T>
inline void _maybe_resize_storage_cpu(T* self, int64_t new_size_bytes) {
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->storage();

  if (!storage) {
    Storage new_storage(new MaybeOwningStorage(new_size_bytes, self->device()));
    self->set_storage(std::move(new_storage));
  } else if (new_size_bytes > static_cast<int64_t>(storage->nbytes())) {
    resize_bytes_cpu(storage.get(), new_size_bytes);
  }
}

template <typename T>
inline T* _resize_impl_(
    T* self,
    c10::IntArrayRef size,
    std::optional<c10::IntArrayRef> stride,
    bool resize_storage) {
  if (self->sizes() == size && (!stride || self->strides() == stride.value())) {
    return self;
  }

  const auto itemsize = c10::elementSize(self->dtype());
  const auto storage_offset = self->storage_offset();
  int64_t storage_size = 1;

  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size =
        compute_storage_nbytes(size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size =
        compute_storage_nbytes_contiguous(size, itemsize, storage_offset);
  }

  if (resize_storage) {
    _maybe_resize_storage_cpu(self, storage_size);
  }

  return self;
}

template <typename T>
inline const T& _resize_(
    const T& self,
    c10::IntArrayRef size,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  T* self_ = const_cast<T*>(&self);
  _resize_impl_<T>(self_, size, /*stride=*/std::nullopt, true);

  if (optional_memory_format.has_value()) {
    c10::MemoryFormat memory_format =
        static_cast<c10::MemoryFormat>(optional_memory_format.value());
    TORCH_CHECK(
        memory_format != c10::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }

  return self;
}

} // namespace torch::standalone
