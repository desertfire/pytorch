#pragma once

// This header mimics APIs in aoti_torch/c/shim.h in a libtorch-free way.
//
#include <torch/csrc/inductor/aoti_libtorch_free/device_type.h>
#include <torch/csrc/inductor/aoti_libtorch_free/layout.h>
#include <torch/csrc/inductor/aoti_libtorch_free/slim_tensor.h>

using AtenTensorOpaque = aoti::libtorch_free::SlimTensor;
using AtenTensorHandle = aoti::libtorch_free::SlimTensor*;

// AOTIProxyExecutorHandle isn't supported in libtorch-free mode.
// Just defining it to void* to make the code compile
using AOTIProxyExecutorHandle = void*;

#ifdef __cplusplus
extern "C" {
#endif
bool aoti_torch_grad_mode_is_enabled();

void aoti_torch_grad_mode_set_enabled(bool enabled);

int32_t aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

int32_t aoti_torch_get_data_ptr(AtenTensorHandle tensor, void** ret_data_ptr);

int32_t aoti_torch_get_sizes(AtenTensorHandle tensor, int64_t** ret_sizes);

AOTITorchError aoti_torch_get_size(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_size);

int32_t aoti_torch_get_strides(AtenTensorHandle tensor, int64_t** ret_strides);

AOTITorchError aoti_torch_get_stride(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_stride);

AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size);

int32_t aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset);

int32_t aoti_torch_get_device_type(
    AtenTensorHandle tensor,
    int32_t* ret_device_type);

int32_t aoti_torch_get_device_index(
    AtenTensorHandle tensor,
    int32_t* ret_device_index);

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor);

AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t offset_increment,
    AtenTensorHandle* ret_new_tensor);

AOTITorchError aoti_torch_as_strided(
    AtenTensorHandle self,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret);

AOTITorchError aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret);

AOTITorchError aoti_torch_clone_preserve_strides(
    AtenTensorHandle self,
    AtenTensorHandle* ret);

#ifdef __cplusplus
} // extern "C"
#endif
