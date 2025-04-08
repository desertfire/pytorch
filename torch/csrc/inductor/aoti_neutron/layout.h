#pragma once
#include <cstdint>

// Similar to c10/core/Layout.h for convenience but doesn't guarantee to always
// be in sync with it

#define FORALL_STRIDE_TYPES(_) \
  _(strided) /* 0 */           \
  _(sparse) /* 1 */            \
  _(sparse_csr) /* 2 */        \
  _(mkldnn) /* 3 */            \
  _(sparse_csc) /* 4 */        \
  _(sparse_bsr) /* 5 */        \
  _(sparse_bsc) /* 6 */        \
  _(jagged) /* 7 */

namespace torch::neutron {
enum class StrideType : int32_t {
#define DEFINE_ENUM(value) value,
  FORALL_STRIDE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};
} // namespace torch::neutron

#ifdef __cplusplus
extern "C" {
#endif

// Define all the aoti_torch_stride_type_*() functions.
#define DEFINE_ENUM(value)                             \
  inline int32_t aoti_torch_layout_##value() {         \
    return (int32_t)torch::neutron::StrideType::value; \
  }
FORALL_STRIDE_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM

#ifdef __cplusplus
} // extern "C"
#endif
