#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/csrc/inductor/aoti_standalone/cpu/pad.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/pad.h>
#include <torch/csrc/inductor/aoti_standalone/factory.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>
#include <torch/torch.h>

using ::testing::ElementsAreArray;
using torch::standalone::SlimTensor;

TEST(PadTest, ConstantPadCPU) {
  at::Tensor at_tensor = at::ones({2, 2}, at::kFloat);
  SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
      at_tensor.data_ptr(),
      c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
      c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
      at::kFloat);

  std::vector<int64_t> pad_vec = {1, 1, 2, 0};
  const char* mode = "constant";
  double value = 3.14;

  AtenTensorHandle result_handle = nullptr;
  AOTITorchError err = aoti_torch_cpu_pad(
      reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
      pad_vec.data(),
      pad_vec.size(),
      mode,
      &value,
      &result_handle);
  ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

  SlimTensor* slim_result = reinterpret_cast<SlimTensor*>(result_handle);

  at::Tensor at_result = at::constant_pad_nd(at_tensor, pad_vec, value);

  // verify
  ASSERT_NE(slim_result, nullptr);
  EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));

  // also verify the data itself is correct
  at::Tensor slim_result_aten_view = at::from_blob(
      slim_result->data_ptr(),
      slim_result->sizes(),
      slim_result->strides(),
      at::kFloat);

  // Debug output
  std::cout << "Expected result (ATen):\n" << at_result << std::endl;
  std::cout << "Actual result (SlimTensor):\n" << slim_result_aten_view << std::endl;
  std::cout << "Expected sizes: ";
  for (auto s : at_result.sizes()) std::cout << s << " ";
  std::cout << "\nActual sizes: ";
  for (auto s : slim_result->sizes()) std::cout << s << " ";
  std::cout << std::endl;

  ASSERT_TRUE(at::equal(slim_result_aten_view, at_result));

  delete slim_result;
}

// #if defined(USE_CUDA)
// TEST(PadTest, ConstantPadCUDA) {

//   if (!torch::cuda::is_available()) {
//     GTEST_SKIP() << "CUDA not available, skipping test";
//   }
//   at::Device cuda_device(at::kCUDA);
//   auto options = at::TensorOptions().dtype(at::kFloat).device(cuda_device);

//   at::Tensor at_tensor = at::ones({2, 2}, options);
//   SlimTensor slim_tensor_self = torch::standalone::create_tensor_from_blob(
//     at_tensor.data_ptr(),
//     c10::IntArrayRef(at_tensor.sizes().data(), at_tensor.dim()),
//     c10::IntArrayRef(at_tensor.strides().data(), at_tensor.dim()),
//     at::kFloat,
//     cuda_device
//   );

//   std::vector<int64_t> pad_vec = {1, 1, 2, 0};
//   const char* mode = "constant";
//   double value = 3.14;

//   AtenTensorHandle result_handle = nullptr;
//   AOTITorchError err = aoti_torch_cuda_pad(
//     reinterpret_cast<AtenTensorHandle>(&slim_tensor_self),
//     pad_vec.data(), pad_vec.size(), mode, &value, &result_handle
//   );
//   ASSERT_EQ(err, AOTI_TORCH_SUCCESS);

//   SlimTensor* slim_result = reinterpret_cast<SlimTensor*>(result_handle);

//   at::Tensor at_result = at::constant_pad_nd(at_tensor, pad_vec, value);

//   // verify
//   ASSERT_NE(slim_result, nullptr);
//   EXPECT_TRUE(slim_result->device().is_cuda());
//   EXPECT_THAT(slim_result->sizes(), ElementsAreArray(at_result.sizes()));

//   // to verify data, copy both results back to the CPU
//   at::Tensor slim_result_cpu = slim_result->to(at::kCPU);
//   at::Tensor at_result_cpu = at_result.to(at::kCPU);

//   at::Tensor slim_result_aten_view = at::from_blob(
//     slim_result_cpu.data_ptr(), slim_result_cpu.sizes(),
//     slim_result_cpu.strides(), at::kFloat
//   );
//   ASSERT_TRUE(at::equal(slim_result_aten_view, at_result_cpu));

//   delete slim_result;

// }
// #endif // USE_CUDA
