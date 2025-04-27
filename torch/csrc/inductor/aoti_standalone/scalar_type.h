#pragma once
#include <array>
#include <cstdint>

// Similar to c10/core/ScalarType.h for convenience but doesn't guarantee to
// always be in sync with it
// clang-format off
#define FORALL_SCALAR_TYPES(A, B)  \
  A(uint8) B(1) /* 0 */            \
  A(int8) B(1) /* 1 */             \
  A(int16) B(2) /* 2 */            \
  A(int32) B(4) /* 3 */            \
  A(int64) B(8) /* 4 */            \
  A(float16) B(2) /* 5 */          \
  A(float32) B(4) /* 6 */          \
  A(float64) B(8) /* 7 */          \
  A(complex32) B(4) /* 8 */        \
  A(complex64) B(8) /* 9 */        \
  A(complex128) B(16) /* 10 */     \
  A(bool) B(1) /* 11 */            \
  A(qint8) B(1) /* 12 */           \
  A(quint8) B(1) /* 13 */          \
  A(qint32) B(4) /* 14 */          \
  A(bfloat16) B(2) /* 15 */        \
  A(quint4x2) B(1) /* 16 */        \
  A(quint2x4) B(1) /* 17 */        \
  A(bits1x8) B(1) /* 18 */         \
  A(bits2x4) B(1) /* 19 */         \
  A(bits4x2) B(1) /* 20 */         \
  A(bits8) B(1) /* 21 */           \
  A(bits16) B(2) /* 22 */          \
  A(float8_e5m2) B(1) /* 23 */     \
  A(float8_e4m3fn) B(1) /* 24 */   \
  A(float8_e5m2fnuz) B(1) /* 25 */ \
  A(float8_e4m3fnuz) B(1) /* 26 */ \
  A(uint16) B(2) /* 27 */          \
  A(uint32) B(4) /* 28 */          \
  A(uint64) B(8) /* 29 */          \
  A(uint1) B(1) /* 30 */           \
  A(uint2) B(1) /* 31 */           \
  A(uint3) B(1) /* 32 */           \
  A(uint4) B(1) /* 33 */           \
  A(uint5) B(1) /* 34 */           \
  A(uint6) B(1) /* 35 */           \
  A(uint7) B(1) /* 36 */           \
  A(int1) B(1) /* 37 */            \
  A(int2) B(1) /* 38 */            \
  A(int3) B(1) /* 39 */            \
  A(int4) B(1) /* 40 */            \
  A(int5) B(1) /* 41 */            \
  A(int6) B(1) /* 42 */            \
  A(int7) B(1) /* 43 */
// clang-format on

namespace torch::native::standalone {
enum class ScalarType : int8_t {
#define DEFINE_A(value) _##value,
#define DEFINE_B(value)
  FORALL_SCALAR_TYPES(DEFINE_A, DEFINE_B)
#undef DEFINE_A
#undef DEFINE_B
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr int32_t SCALAR_TYPE_TO_BYTESIZE[] = {
#define DEFINE_A(value)
#define DEFINE_B(value) value,
    FORALL_SCALAR_TYPES(DEFINE_A, DEFINE_B)
#undef DEFINE_A
#undef DEFINE_B
};
} // namespace torch::native::standalone

#ifdef __cplusplus
extern "C" {
#endif

// Define all the aoti_torch_dtype_*() functions.
#define DEFINE_A(value)                                           \
  inline int32_t aoti_torch_dtype_##value() {                     \
    return (int32_t)torch::native::standalone::ScalarType::_##value; \
  }
#define DEFINE_B(value)
FORALL_SCALAR_TYPES(DEFINE_A, DEFINE_B)
#undef DEFINE_A
#undef DEFINE_B

#ifdef __cplusplus
} // extern "C"
#endif
