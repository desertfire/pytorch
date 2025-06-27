#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

using ::testing::ElementsAreArray;

TEST(SlimTensorInternalTest, SetSizesContiguous) {
  torch::standalone::SlimTensor tensor =
      torch::standalone::create_empty_tensor({}, {}, c10::kFloat);
  EXPECT_EQ(tensor.numel(), 1);

  std::vector<int64_t> new_size_vec = {2, 3, 4};
  c10::IntArrayRef new_sizes(new_size_vec.data(), new_size_vec.size());
  tensor.set_sizes_contiguous(new_sizes);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);

  EXPECT_THAT(tensor.sizes(), ElementsAreArray({2, 3, 4}));
  EXPECT_THAT(tensor.strides(), ElementsAreArray({12, 4, 1}));
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(SlimTensorInternalTest, SetSizesAndStridesNonContiguous) {
  torch::standalone::SlimTensor tensor =
      torch::standalone::create_empty_tensor({}, {}, c10::kFloat);

  // Set sizes and strides to represent a transposed tensor.
  // This is equivalent to a (2, 4) tensor that has been transposed to (4, 2).
  std::vector<int64_t> new_size_vec = {4, 2};

  // Strides of original (2, 4) were {4, 1}
  std::vector<int64_t> new_stride_vec = {1, 4};
  c10::IntArrayRef new_sizes(new_size_vec.data(), new_size_vec.size());
  c10::IntArrayRef new_strides(new_stride_vec.data(), new_stride_vec.size());

  tensor.set_sizes_and_strides(new_sizes, new_strides);

  // verify
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.numel(), 8);
  EXPECT_THAT(tensor.sizes(), ElementsAreArray({4, 2}));
  EXPECT_THAT(tensor.strides(), ElementsAreArray({1, 4}));

  // A transposed tensor is not contiguous.
  EXPECT_FALSE(tensor.is_contiguous());
}

TEST(SlimTensorInternalTest, EmptyTensorRestride) {
  torch::standalone::SlimTensor tensor =
      torch::standalone::create_empty_tensor({}, {}, c10::kFloat);
  std::vector<int64_t> size_vec = {4, 2};
  std::vector<int64_t> stride_vec = {1, 4}; // Non-contiguous strides
  tensor.set_sizes_and_strides(
      c10::IntArrayRef(size_vec.data(), size_vec.size()),
      c10::IntArrayRef(stride_vec.data(), stride_vec.size()));
  // it shouldn't be contiguous first
  EXPECT_FALSE(tensor.is_contiguous());

  // make the tensor contiguous.
  tensor.empty_tensor_restride(c10::MemoryFormat::Contiguous);

  // Sizes should NOT have changed.
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_THAT(tensor.sizes(), ElementsAreArray({4, 2}));

  // Strides SHOULD have changed to be contiguous for a (4, 2) tensor.
  EXPECT_THAT(tensor.strides(), ElementsAreArray({2, 1}));

  // The contiguity flag should now be true.
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(SlimTensorInternalTest, CopyFromContiguous) {
  // Create a contiguous source tensor.
  std::vector<float> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  torch::standalone::SlimTensor src_tensor =
      torch::standalone::create_tensor_from_blob(
          src_data.data(), {2, 4}, {4, 1}, c10::kFloat);
  ASSERT_TRUE(src_tensor.is_contiguous());

  // Create an empty, contiguous destination tensor.
  std::vector<int64_t> dst_strides = {4, 1};
  torch::standalone::SlimTensor dst_tensor =
      torch::standalone::create_empty_tensor({2, 4}, dst_strides, c10::kFloat);
  ASSERT_TRUE(dst_tensor.is_contiguous());

  dst_tensor.copy_(src_tensor);

  // When we verify the destination tensor's data should be an exact copy of the
  // source.
  float* dst_data_ptr = static_cast<float*>(dst_tensor.data_ptr());

  // Compare the actual data in the destination tensor with the source data.
  for (size_t i = 0; i < src_data.size(); ++i) {
    EXPECT_EQ(dst_data_ptr[i], src_data[i]);
  }
}

TEST(SlimTensorInternalTest, CopyFromNonContiguous) {
  // Create a non-contiguous source tensor by setting transposed metadata.
  // This simulates a (2, 4) tensor with data [0, 1, 2, 3, 4, 5, 6, 7]
  // that has been transposed to (4, 2).
  std::vector<float> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  torch::standalone::SlimTensor src_tensor =
      torch::standalone::create_tensor_from_blob(
          src_data.data(), {4, 2}, {1, 4}, c10::kFloat);
  ASSERT_FALSE(src_tensor.is_contiguous());

  // Create an empty, contiguous destination tensor of the same shape.
  std::vector<int64_t> dst_strides = {2, 1};
  torch::standalone::SlimTensor dst_tensor =
      torch::standalone::create_empty_tensor({4, 2}, dst_strides, c10::kFloat);
  ASSERT_TRUE(dst_tensor.is_contiguous());

  // Perform the copy.
  dst_tensor.copy_(src_tensor);

  // When we verify the destination tensor should remain contiguous.
  EXPECT_TRUE(dst_tensor.is_contiguous());

  // The logical data of the transposed source tensor is:
  // [[0, 4],
  //  [1, 5],
  //  [2, 6],
  //  [3, 7]]
  //
  // When copied to a contiguous layout, the memory should be:
  // [0, 4, 1, 5, 2, 6, 3, 7]
  std::vector<float> expected_data = {0, 4, 1, 5, 2, 6, 3, 7};
  float* dst_data_ptr = static_cast<float*>(dst_tensor.data_ptr());

  // Compare the actual data in the destination tensor with the expected layout.
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_EQ(dst_data_ptr[i], expected_data[i]);
  }
}
