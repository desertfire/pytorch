#include <c10/core/ScalarType.h>
#include <c10/util/Array.h>
#include <array>

namespace c10 {
std::pair<std::string, std::string> getDtypeNames(c10::ScalarType scalarType) {
  switch (scalarType) {
    case c10::ScalarType::UInt1:
      return std::make_pair("uint1", "bit");
    case c10::ScalarType::UInt2:
      return std::make_pair("uint2", "");
    case c10::ScalarType::UInt3:
      return std::make_pair("uint3", "");
    case c10::ScalarType::UInt4:
      return std::make_pair("uint4", "");
    case c10::ScalarType::UInt5:
      return std::make_pair("uint5", "");
    case c10::ScalarType::UInt6:
      return std::make_pair("uint6", "");
    case c10::ScalarType::UInt7:
      return std::make_pair("uint7", "");
    case c10::ScalarType::Byte:
      // no "byte" because byte is signed in numpy and we overload
      // byte to mean bool often
      return std::make_pair("uint8", "");
    case c10::ScalarType::UInt16:
      return std::make_pair("uint16", "");
    case c10::ScalarType::UInt32:
      return std::make_pair("uint32", "");
    case c10::ScalarType::UInt64:
      return std::make_pair("uint64", "");
    case c10::ScalarType::Int1:
      return std::make_pair("int1", "");
    case c10::ScalarType::Int2:
      return std::make_pair("int2", "");
    case c10::ScalarType::Int3:
      return std::make_pair("int3", "");
    case c10::ScalarType::Int4:
      return std::make_pair("int4", "");
    case c10::ScalarType::Int5:
      return std::make_pair("int5", "");
    case c10::ScalarType::Int6:
      return std::make_pair("int6", "");
    case c10::ScalarType::Int7:
      return std::make_pair("int7", "");
    case c10::ScalarType::Char:
      // no "char" because it is not consistently signed or unsigned; we want
      // to move to int8
      return std::make_pair("int8", "");
    case c10::ScalarType::Double:
      return std::make_pair("float64", "double");
    case c10::ScalarType::Float:
      return std::make_pair("float32", "float");
    case c10::ScalarType::Int:
      return std::make_pair("int32", "int");
    case c10::ScalarType::Long:
      return std::make_pair("int64", "long");
    case c10::ScalarType::Short:
      return std::make_pair("int16", "short");
    case c10::ScalarType::Half:
      return std::make_pair("float16", "half");
    case c10::ScalarType::ComplexHalf:
      return std::make_pair("complex32", "chalf");
    case c10::ScalarType::ComplexFloat:
      return std::make_pair("complex64", "cfloat");
    case c10::ScalarType::ComplexDouble:
      return std::make_pair("complex128", "cdouble");
    case c10::ScalarType::Bool:
      return std::make_pair("bool", "");
    case c10::ScalarType::QInt8:
      return std::make_pair("qint8", "");
    case c10::ScalarType::QUInt8:
      return std::make_pair("quint8", "");
    case c10::ScalarType::QInt32:
      return std::make_pair("qint32", "");
    case c10::ScalarType::BFloat16:
      return std::make_pair("bfloat16", "");
    case c10::ScalarType::QUInt4x2:
      return std::make_pair("quint4x2", "");
    case c10::ScalarType::QUInt2x4:
      return std::make_pair("quint2x4", "");
    case c10::ScalarType::Bits1x8:
      return std::make_pair("bits1x8", "");
    case c10::ScalarType::Bits2x4:
      return std::make_pair("bits2x4", "");
    case c10::ScalarType::Bits4x2:
      return std::make_pair("bits4x2", "");
    case c10::ScalarType::Bits8:
      return std::make_pair("bits8", "");
    case c10::ScalarType::Bits16:
      return std::make_pair("bits16", "");
    case c10::ScalarType::Float8_e5m2:
      return std::make_pair("float8_e5m2", "");
    case c10::ScalarType::Float8_e4m3fn:
      return std::make_pair("float8_e4m3fn", "");
    case c10::ScalarType::Float8_e5m2fnuz:
      return std::make_pair("float8_e5m2fnuz", "");
    case c10::ScalarType::Float8_e4m3fnuz:
      return std::make_pair("float8_e4m3fnuz", "");
    case c10::ScalarType::Float8_e8m0fnu:
      // TODO(#146647): macroify all of this
      return std::make_pair("float8_e8m0fnu", "");
    case c10::ScalarType::Float4_e2m1fn_x2:
      return std::make_pair("float4_e2m1fn_x2", "");
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap() {
  static std::unordered_map<std::string, ScalarType> result;
  if (!result.empty()) {
    return result;
  }

#define DEFINE_SCALAR_TYPE(_1, n) c10::ScalarType::n,

  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

#undef DEFINE_SCALAR_TYPE

  for (auto scalar_type : all_scalar_types) {
    auto names = getDtypeNames(scalar_type);
    result[std::get<0>(names)] = scalar_type;
    if (!std::get<1>(names).empty()) {
      result[std::get<1>(names)] = scalar_type;
    }
  }
  return result;
}

} // namespace c10
