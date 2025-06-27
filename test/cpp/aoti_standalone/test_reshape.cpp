#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/reshape.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

using ::testing::ElementsAreArray;

// Test Case 1: Reshape on a contiguous tensor that can be performed as a view.
TEST(ReshapeTest, ReshapeAsView) {
  // 1. Setup
  at::Tensor at_tensor = at::arange(12, at::kFloat).reshape({3, 4});
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat);
  ASSERT_TRUE(slim_tensor_self.is_contiguous());

  // 2. Action
  std::vector<int64_t> new_shape_vec = {2, 6};
  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err = aoti_torch_cpu_reshape(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      new_shape_vec.data(),
      new_shape_vec.size(),
      &result_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  // For comparison, get the ground-truth result from ATen
  at::Tensor at_result =
      at::reshape(at_tensor, c10::IntArrayRef(new_shape_vec));

  // 3. Verify
  ASSERT_NE(slim_result, nullptr);
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));
  EXPECT_THAT(slim_result->strides(), ElementsAreArray(at_result.strides()));
  EXPECT_TRUE(slim_result->is_contiguous()); // Reshaping a contiguous tensor
                                             // should remain contiguous

  // CRITICAL: Verify it's a view by checking that the data pointer is unchanged
  EXPECT_EQ(slim_result->data_ptr(), at_tensor.data_ptr());

  delete slim_result;
}

// Test Case 2: Reshape on a non-contiguous tensor, which should trigger a copy.
TEST(ReshapeTest, ReshapeWithCopy) {
  // 1. Setup: Create a non-contiguous tensor by transposing it.
  at::Tensor at_tensor_original = at::arange(12, at::kFloat).reshape({3, 4});
  at::Tensor at_tensor_transposed = at::transpose(at_tensor_original, 0, 1);
  ASSERT_FALSE(at_tensor_transposed.is_contiguous());

  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor_transposed.data_ptr(),
          c10::IntArrayRef(
              at_tensor_transposed.sizes().data(), at_tensor_transposed.dim()),
          c10::IntArrayRef(
              at_tensor_transposed.strides().data(),
              at_tensor_transposed.dim()),
          at::kFloat);

  // 2. Action: Reshape the non-contiguous tensor. This cannot be a view.
  std::vector<int64_t> new_shape_vec = {2, 6};
  AtenTensorHandle result_handle = nullptr;

  AOTITorchError err = aoti_torch_cpu_reshape(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      new_shape_vec.data(),
      new_shape_vec.size(),
      &result_handle);

  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  at::Tensor at_result =
      at::reshape(at_tensor_transposed, c10::IntArrayRef(new_shape_vec));

  // 3. Verify
  ASSERT_NE(slim_result, nullptr);
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));
  EXPECT_TRUE(slim_result->is_contiguous());

  // CRITICAL: Verify a copy was made by checking that the data pointer is
  // DIFFERENT
  EXPECT_NE(slim_result->data_ptr(), at_tensor_original.data_ptr());

  // Verify the data itself is correct by comparing to the ATen result
  at::Tensor slim_result_aten_view = at::from_blob(
      slim_result->data_ptr(),
      slim_result->sizes(),
      slim_result->strides(),
      at::kFloat);
  ASSERT_TRUE(at::equal(slim_result_aten_view, at_result));

  delete slim_result;
}

// Test Case 3: Reshape with an inferred dimension (-1).
TEST(ReshapeTest, ReshapeWithInferredDimension) {
  // 1. Setup
  at::Tensor at_tensor = at::arange(12, at::kFloat).reshape({3, 4});
  torch::standalone::SlimTensor slim_tensor_self =
      torch::standalone::create_tensor_from_blob(
          at_tensor.data_ptr(),
          c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
          c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
          at::kFloat);

  // 2. Action: Propose a shape with -1
  std::vector<int64_t> new_shape_vec = {2, -1, 3};
  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err = aoti_torch_cpu_reshape(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      new_shape_vec.data(),
      new_shape_vec.size(),
      &result_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);
  torch::standalone::SlimTensor* slim_result =
      reinterpret_cast<torch::standalone::SlimTensor*>(result_handle);

  // 3. Verify: The inferred shape should be {2, 2, 3}
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray({2, 2, 3}));
  EXPECT_EQ(
      slim_result->data_ptr(), at_tensor.data_ptr()); // Should still be a view

  delete slim_result;
}
