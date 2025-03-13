#pragma once
#include <cstdint>

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

namespace aoti::libtorch_free {
using DeviceIndex = int32_t;

enum class DeviceType : int8_t {
#define DEFINE_ENUM(value) value,
  FORALL_DEVICE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};
} // namespace aoti::libtorch_free

#ifdef __cplusplus
extern "C" {
#endif

// Define all the aoti_torch_device_type_*() functions.
#define DEFINE_ENUM(value)                                  \
  inline int32_t aoti_torch_device_type_##value() {         \
    return (int32_t)aoti::libtorch_free::DeviceType::value; \
  }
FORALL_DEVICE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM

#ifdef __cplusplus
} // extern "C"
#endif
