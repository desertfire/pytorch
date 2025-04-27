#pragma once
#include <cstdint>
#include <string>

// Similar to c10/core/DeviceType.h for convenience but doesn't guarantee to
// always be in sync with it
#define FORALL_DEVICE_TYPES(_) \
  _(cpu) /* 0 */               \
  _(cuda) /* 1 */              \
  _(mkldnn) /* 2 */            \
  _(opengl) /* 3 */            \
  _(opencl) /* 4 */            \
  _(ideep) /* 5 */             \
  _(hip) /* 6 */               \
  _(fpga) /* 7 */              \
  _(maia) /* 8 */              \
  _(xla) /* 9 */               \
  _(vulkan) /* 10 */           \
  _(metal) /* 11 */            \
  _(xpu) /* 12 */              \
  _(mps) /* 13 */              \
  _(meta) /* 14 */             \
  _(hpu) /* 15 */              \
  _(ve) /* 16 */               \
  _(lazy) /* 17 */             \
  _(ipu) /* 18 */              \
  _(mtia) /* 19 */             \
  _(privateuse1) /* 20 */

namespace torch::native::standalone {
using DeviceIndex = int32_t;

enum class DeviceType : int8_t {
#define DEFINE_ENUM(value) value,
  FORALL_DEVICE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr const char* DEVICE_TYPE_TO_STR[] = {
#define DEFINE_ENUM(value) #value,
    FORALL_DEVICE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Device struct that encapsulates both device type and device index
struct Device {
  DeviceType type_;
  DeviceIndex index_;

  Device() : type_(DeviceType::cpu), index_(0) {}
  Device(DeviceType t, DeviceIndex i) : type_(t), index_(i) {}

  bool is_cpu() const {
    return type_ == DeviceType::cpu;
  }
  bool is_cuda() const {
    return type_ == DeviceType::cuda;
  }

  // Equality operators
  bool operator==(const Device& other) const {
    return type_ == other.type_ && index_ == other.index_;
  }

  bool operator!=(const Device& other) const {
    return !(*this == other);
  }

  std::string str() const {
    return std::string(DEVICE_TYPE_TO_STR[static_cast<size_t>(type_)]) +
        std::string(":") + std::to_string(index_);
  }
};

const Device CPU_DEVICE = Device(DeviceType::cpu, 0);
} // namespace torch::native::standalone

#ifdef __cplusplus
extern "C" {
#endif

// Define all the aoti_torch_device_type_*() functions.
#define DEFINE_ENUM(value)                                     \
  inline int32_t aoti_torch_device_type_##value() {            \
    return (int32_t)torch::native::standalone::DeviceType::value; \
  }
FORALL_DEVICE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM

#ifdef __cplusplus
} // extern "C"
#endif
