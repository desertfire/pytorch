#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/pad.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/pad.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

using ::testing::ElementsAreArray;
using torch::standalone::SlimTensor;

void run_and_verify_pad(
    SlimTensor& slim_input,
    const at::Tensor& at_input,
    const std::vector<int64_t>& pad_vec,
    double value) {
  const char* mode = "constant";
  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err;

  if (slim_input.device().is_cpu()) {
    err = aoti_torch_cpu_pad(
        reinterpret_cast<AtenTensorHandle>(&slim_input),
        pad_vec.data(),
        pad_vec.size(),
        mode,
        &value,
        &result_handle);
  } else {
#if defined(USE_CUDA)
    err = aoti_torch_cuda_pad(
        reinterpret_cast<AtenTensorHandle>(&slim_input),
        pad_vec.data(),
        pad_vec.size(),
        mode,
        &value,
        &result_handle);
#endif
  }
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  SlimTensor* slim_result = reinterpret_cast<SlimTensor*>(result_handle);
  at::Tensor at_result = at::constant_pad_nd(at_input, pad_vec, value);

  // verify
  ASSERT_NE(slim_result, nullptr);
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));

  // to verify data copy both results to CPU
  SlimTensor slim_result_cpu = slim_result->to(at::kCPU);
  at::Tensor at_result_cpu = at_result.to(at::kCPU);

  at::Tensor slim_result_aten_view = at::from_blob(
      slim_result_cpu.data_ptr(),
      slim_result_cpu.sizes(),
      slim_result_cpu.strides(),
      slim_result_cpu.dtype());

  // Debug output for negative padding test
  if (pad_vec.size() == 4 && pad_vec[0] == -1 && pad_vec[1] == -1 &&
      pad_vec[2] == -1 && pad_vec[3] == -1) {
    std::cout << "=== DEBUG: NegativePadding Test ===" << std::endl;
    std::cout << "Input tensor shape: ";
    for (auto s : at_input.sizes())
      std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "Expected result shape: ";
    for (auto s : at_result_cpu.sizes())
      std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "Actual result shape: ";
    for (auto s : slim_result_cpu.sizes())
      std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "Expected result:\n" << at_result_cpu << std::endl;
    std::cout << "Actual result:\n" << slim_result_aten_view << std::endl;
    std::cout << "===================================" << std::endl;
  }

  ASSERT_TRUE(at::equal(slim_result_aten_view, at_result_cpu));

  delete slim_result;
}

TEST(PadTest, ConstantPadCPU) {
  at::Tensor at_tensor = at::ones({2, 2}, at::kFloat);
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(), at_tensor.sizes(), at_tensor.strides(), at::kFloat);
  run_and_verify_pad(slim_tensor_self, at_tensor, {1, 1, 2, 0}, 3.14);
}

TEST(PadTest, NegativePaddingCPU) {
  at::Tensor at_tensor = at::arange(12, at::kFloat).reshape({3, 4});
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(), at_tensor.sizes(), at_tensor.strides(), at::kFloat);
  // Slice one element from each side of each dimension
  run_and_verify_pad(slim_tensor_self, at_tensor, {-1, -1, -1, -1}, 0.0);
}

TEST(PadTest, MixedPaddingCPU) {
  at::Tensor at_tensor = at::ones({3, 3}, at::kFloat);
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(), at_tensor.sizes(), at_tensor.strides(), at::kFloat);
  // Pad last dim: left=1, right=-1 (slice). Pad first dim: top=0, bottom=2
  run_and_verify_pad(slim_tensor_self, at_tensor, {1, -1, 0, 2}, 5.0);
}

TEST(PadTest, PadFewerDimsCPU) {
  at::Tensor at_tensor = at::ones({2, 3, 4}, at::kFloat);
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(), at_tensor.sizes(), at_tensor.strides(), at::kFloat);
  // Only pad the last dimension
  run_and_verify_pad(slim_tensor_self, at_tensor, {1, 1}, 0.0);
}

#if defined(USE_CUDA)
TEST(PadTest, ConstantPadCUDA) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA not available";
  at::Device cuda_device(at::kCUDA);
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device);

  at::Tensor at_tensor = at::ones({2, 2}, options);
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(),
      at_tensor.sizes(),
      at_tensor.strides(),
      at::kFloat,
      cuda_device);
  run_and_verify_pad(slim_tensor_self, at_tensor, {1, 1, 2, 0}, 3.14);
}

TEST(PadTest, NegativePaddingCUDA) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA not available";
  at::Device cuda_device(at::kCUDA);
  auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device);

  at::Tensor at_tensor = at::arange(12, options).reshape({3, 4});
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(),
      at_tensor.sizes(),
      at_tensor.strides(),
      at::kFloat,
      cuda_device);
  run_and_verify_pad(slim_tensor_self, at_tensor, {-1, -1, -1, -1}, 0.0);
}
#endif // USE_CUDA
