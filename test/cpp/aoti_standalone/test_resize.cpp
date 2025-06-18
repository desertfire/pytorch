#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/resize.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/resize.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

namespace torch::standalone {

using ::testing::ElementsAreArray;

// Test case 1: Resizing a tensor to a smaller size.
// This should modify metadata but NOT reallocate the underlying data buffer.
TEST(SlimTensorTest, ResizeOpShrinkCPU) {
  at::Tensor at_tensor = at::arange(12, at::kFloat).reshape({3, 4});
  at::Tensor at_tensor_for_resize = at_tensor.clone();

  // create an owning SlimTensor and copy the data into it
  SlimTensor slim_tensor = create_empty_tensor(
      {at_tensor.sizes().data(), static_cast<size_t>(at_tensor.dim())},
      {at_tensor.strides().data(), static_cast<size_t>(at_tensor.dim())},
      at_tensor.scalar_type());
  slim_tensor.copy_(create_tensor_from_blob(
      at_tensor.data_ptr(),
      {at_tensor.sizes().data(), static_cast<size_t>(at_tensor.dim())},
      {at_tensor.strides().data(), static_cast<size_t>(at_tensor.dim())},
      at_tensor.scalar_type()));
  void* original_data_ptr = slim_tensor.data_ptr();

  // perform the resize operation on both tensors
  std::vector<int64_t> new_size = {2, 3};

  AOTITorchError err = aoti_torch_cpu_resize_(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor),
      new_size.data(),
      new_size.size(),
      nullptr // No memory format change
  );
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  // also do it for aten tensor
  at_tensor_for_resize.resize_({2, 3});

  // 4. Verify results
  EXPECT_EQ(slim_tensor.dim(), 2);
  EXPECT_THAT(slim_tensor.sizes(), ElementsAreArray({2, 3}));
  EXPECT_THAT(slim_tensor.strides(), ElementsAreArray({3, 1}));
  EXPECT_EQ(slim_tensor.numel(), 6);

  // For shrinking, the data pointer should NOT change
  EXPECT_EQ(slim_tensor.data_ptr(), original_data_ptr);

  // Check that the data within the new view is still correct
  float* slim_data = static_cast<float*>(slim_tensor.data_ptr());
  float* at_data = static_cast<float*>(at_tensor_for_resize.data_ptr());
  for (size_t i = 0; i < slim_tensor.numel(); i++) {
    // Original data should be preserved
    EXPECT_FLOAT_EQ(slim_data[i], at_data[i]);
    EXPECT_FLOAT_EQ(slim_data[i], static_cast<float>(i));
  }
}

// Test case 2: Resizing a tensor to a larger size for CPU.
// This MUST reallocate the underlying data buffer.
TEST(SlimTensorTest, ResizeOpGrowCPU) {
  at::Tensor at_tensor = at::arange(4, at::kFloat).reshape({2, 2});
  at::Tensor at_tensor_for_resize = at_tensor.clone();

  // Create an OWNING SlimTensor by creating a fresh empty tensor
  // and copying the data. This is essential because we can't reallocate
  // the memory of an at::Tensor that we don't own.
  SlimTensor slim_tensor = create_empty_tensor(
      {at_tensor.sizes().data(), (size_t)at_tensor.dim()},
      {at_tensor.strides().data(), (size_t)at_tensor.dim()},
      at_tensor.scalar_type());
  slim_tensor.copy_(create_tensor_from_blob(
      at_tensor.data_ptr(),
      {at_tensor.sizes().data(), static_cast<size_t>(at_tensor.dim())},
      {at_tensor.strides().data(), static_cast<size_t>(at_tensor.dim())},
      at_tensor.scalar_type()));
  void* original_data_ptr = slim_tensor.data_ptr();

  // perform the resize operation on our slim tensor
  std::vector<int64_t> new_size = {3, 3};
  AOTITorchError err = aoti_torch_cpu_resize_(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor),
      new_size.data(),
      new_size.size(),
      nullptr);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  // aten will fill new memory with arbitrary data
  at_tensor_for_resize.resize_({3, 3});

  // verify
  EXPECT_EQ(slim_tensor.dim(), 2);
  EXPECT_THAT(slim_tensor.sizes(), ElementsAreArray({3, 3}));
  EXPECT_EQ(slim_tensor.numel(), 9);

  // For growing, the data pointer MUST change
  EXPECT_NE(slim_tensor.data_ptr(), original_data_ptr);

  // Check that the original data was copied to the new buffer
  float* slim_data = static_cast<float*>(slim_tensor.data_ptr());
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(slim_data[i], static_cast<float>(i));
  }
}

// Test case 3: Resizing a tensor to have 0 elements.
TEST(SlimTensorTest, ResizeOpToZeroCPU) {
  at::Tensor at_tensor = at::ones({2, 2});
  at::Tensor at_tensor_for_resize = at_tensor.clone();

  SlimTensor slim_tensor = create_tensor_from_blob(
      at_tensor.data_ptr(),
      {at_tensor.sizes().data(), (size_t)at_tensor.dim()},
      {at_tensor.strides().data(), (size_t)at_tensor.dim()},
      at_tensor.scalar_type());

  // Perform resize
  std::vector<int64_t> new_size = {2, 0, 2};
  AOTITorchError err = aoti_torch_cpu_resize_(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor),
      new_size.data(),
      new_size.size(),
      nullptr);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  at_tensor_for_resize.resize_({2, 0, 2});

  // verify
  EXPECT_EQ(slim_tensor.dim(), 3);
  EXPECT_THAT(slim_tensor.sizes(), ElementsAreArray({2, 0, 2}));
  EXPECT_EQ(slim_tensor.numel(), 0);
  EXPECT_EQ(at_tensor_for_resize.numel(), 0);
}

// Test case 4: Resizing a tensor to a larger size for CUDA.
// This MUST also reallocate the underlying data buffer.
TEST(SlimTensorTest, ResizeOpGrowCUDA) {
  at::Tensor at_tensor = at::arange(4, at::kFloat).reshape({2, 2});
  at::Tensor at_tensor_for_resize = at_tensor.clone();

  // Create an OWNING SlimTensor by creating a fresh empty tensor
  // and copying the data. This is essential because we can't reallocate
  // the memory of an at::Tensor that we don't own.
  SlimTensor slim_tensor = create_empty_tensor(
      {at_tensor.sizes().data(), (size_t)at_tensor.dim()},
      {at_tensor.strides().data(), (size_t)at_tensor.dim()},
      at_tensor.scalar_type());
  slim_tensor.copy_(create_tensor_from_blob(
      at_tensor.data_ptr(),
      {at_tensor.sizes().data(), static_cast<size_t>(at_tensor.dim())},
      {at_tensor.strides().data(), static_cast<size_t>(at_tensor.dim())},
      at_tensor.scalar_type()));
  void* original_data_ptr = slim_tensor.data_ptr();

  // perform the resize operation on our slim tensor
  std::vector<int64_t> new_size = {3, 3};
  AOTITorchError err = aoti_torch_cuda_resize_(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor),
      new_size.data(),
      new_size.size(),
      nullptr);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  // aten will fill new memory with arbitrary data
  at_tensor_for_resize.resize_({3, 3});

  // verify
  EXPECT_EQ(slim_tensor.dim(), 2);
  EXPECT_THAT(slim_tensor.sizes(), ElementsAreArray({3, 3}));
  EXPECT_EQ(slim_tensor.numel(), 9);

  // For growing, the data pointer MUST change
  EXPECT_NE(slim_tensor.data_ptr(), original_data_ptr);

  // Check that the original data was copied to the new buffer
  float* slim_data = static_cast<float*>(slim_tensor.data_ptr());
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(slim_data[i], static_cast<float>(i));
  }
}

} // namespace torch::standalone
