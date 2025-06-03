#pragma once

#include <torch/standalone/core/ScalarType.h>

namespace c10 {
// NB: despite its generic sounding name, the macros that don't take _AND
// are mostly only used by tensorexpr
#define AT_FORALL_INT_TYPES(_) \
  _(uint8_t, Byte)             \
  _(int8_t, Char)              \
  _(int16_t, Short)            \
  _(int, Int)                  \
  _(int64_t, Long)

#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(double, Double)

// These macros are often controlling how many template instantiations we
// create for kernels.  It is typically inappropriate to add new dtypes here,
// instead, new types should be added to use sites on a case-by-case basis.
// We generally are not accepting new dtypes due to binary size concerns.

#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _) \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE>::t),  \
    SCALARTYPE)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _) \
  _(uint8_t, Byte)                                               \
  _(int8_t, Char)                                                \
  _(int16_t, Short)                                              \
  _(int, Int)                                                    \
  _(int64_t, Long)                                               \
  _(float, Float)                                                \
  _(double, Double)                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE1>::t),                \
    SCALARTYPE1)                                                 \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE2>::t),                \
    SCALARTYPE2)

#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE1>::t),                             \
    SCALARTYPE1)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE2>::t),                             \
    SCALARTYPE2)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE3>::t),                             \
    SCALARTYPE3)

#define AT_FORALL_SCALAR_TYPES_AND7(              \
    SCALARTYPE1,                                  \
    SCALARTYPE2,                                  \
    SCALARTYPE3,                                  \
    SCALARTYPE4,                                  \
    SCALARTYPE5,                                  \
    SCALARTYPE6,                                  \
    SCALARTYPE7,                                  \
    _)                                            \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE5>::t), \
    SCALARTYPE5)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE6>::t), \
    SCALARTYPE6)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE7>::t), \
    SCALARTYPE7)

#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define AT_FORALL_FLOAT8_TYPES(_)         \
  _(at::Float8_e5m2, Float8_e5m2)         \
  _(at::Float8_e4m3fn, Float8_e4m3fn)     \
  _(at::Float8_e5m2fnuz, Float8_e5m2fnuz) \
  _(at::Float8_e4m3fnuz, Float8_e4m3fnuz) \
  _(at::Float8_e8m0fnu, Float8_e8m0fnu)

#define AT_FORALL_COMPLEX_TYPES(_)     \
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

// Returns a pair of strings representing the names for each dtype.
// The returned pair is (name, legacy_name_if_applicable)
C10_API std::pair<std::string, std::string> getDtypeNames(
    c10::ScalarType scalarType);

// Returns a map of string name to dtype.
C10_API const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap();

} // namespace c10
