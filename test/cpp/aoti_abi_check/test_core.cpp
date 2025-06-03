#include <gtest/gtest.h>

#include <torch/standalone/core/Device.h>

namespace torch {
namespace aot_inductor {
using namespace torch::standalone;

TEST(TestCore, TestDeviceType) {
  // clang-format off
  constexpr DeviceType expected_device_types[] = {
    kCPU,
    kCUDA,
    kMKLDNN,
    kOPENGL,
    kOPENCL,
    kIDEEP,
    kHIP,
    kFPGA,
    kMAIA,
    kXLA,
    kVulkan,
    kMetal,
    kXPU,
    kMPS,
    kMeta,
    kHPU,
    kVE,
    kLazy,
    kIPU,
    kMTIA,
    kPrivateUse1,
  };
  // clang-format on
  for (int8_t i = 0;
       i < static_cast<int8_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
       ++i) {
    EXPECT_EQ(static_cast<DeviceType>(i), expected_device_types[i]);
  }
}

TEST(TestCore, PrintDeviceType) {
  for (int8_t i = 0;
       i < static_cast<int8_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
       ++i) {
    std::cout << i << ": DeviceType::" << static_cast<DeviceType>(i)
              << std::endl;
  }
}

TEST(TestCore, TestDevice) {
  Device cpu_device(kCPU);
  EXPECT_EQ(cpu_device.type(), kCPU);
  EXPECT_EQ(cpu_device.index(), -1);
  EXPECT_EQ(cpu_device.str(), "cpu");

  Device cuda_device(kCUDA, 0);
  EXPECT_EQ(cuda_device.type(), kCUDA);
  EXPECT_EQ(cuda_device.index(), 0);
  EXPECT_EQ(cuda_device.str(), "cuda:0");

  Device cpu_device_copy(cpu_device);
  EXPECT_EQ(cpu_device_copy.type(), kCPU);
  EXPECT_EQ(cpu_device_copy.index(), -1);
  EXPECT_EQ(cpu_device_copy.str(), "cpu");

  Device cuda_device_copy(cuda_device);
  EXPECT_EQ(cuda_device_copy.type(), kCUDA);
  EXPECT_EQ(cuda_device_copy.index(), 0);
  EXPECT_EQ(cuda_device_copy.str(), "cuda:0");

  EXPECT_EQ(cuda_device, cuda_device_copy);
  EXPECT_NE(cuda_device, cpu_device_copy);
}

TEST(TestCore, PrintDevice) {
  Device cpu_device(kCPU);
  std::cout << "cpu_device: " << cpu_device << std::endl;

  Device cuda_device(kCUDA, 0);
  std::cout << "cuda_device: " << cuda_device << std::endl;

} // namespace aot_inductor
} // namespace torch
