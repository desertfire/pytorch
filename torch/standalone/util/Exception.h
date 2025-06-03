#pragma once

#include <torch/standalone/macros/Macros.h>

#include <sstream>
#include <string>

#ifdef TORCH_STANDALONE
// In the standalone version, TORCH_CHECK throws std::runtime_error
// instead of c10::Error, because c10::Error transitively calls too
// much code to be implmented as header-only.
// This is useful when AOTInductor generated code uses some PyTorch
// library header-only code directly, such as Vectorized<T>, and
// those code can use TORCH_CHECK to check for errors.

#ifdef STRIP_ERROR_MESSAGES
#define TORCH_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
#define TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    throw std::runtime_error(TORCH_CHECK_MSG( \
        cond,                                 \
        "",                                   \
        __func__,                             \
        ", ",                                 \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", ",                                 \
        __VA_ARGS__));                        \
  }

#else // STRIP_ERROR_MESSAGES
namespace c10::detail {
template <typename... Args>
std::string torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  // This is similar to the one in c10/util/Exception.h, but does
  // not depend on the more complex c10::str() function. ostringstream
  // may support less data types than c10::str(), but should be sufficient
  // in the standalone world.
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}
inline C10_API const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline C10_API const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}
} // namespace c10::detail

#define TORCH_CHECK_MSG(cond, type, ...)                   \
  (::c10::detail::torchCheckMsgImpl(                       \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#define TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    throw std::runtime_error(TORCH_CHECK_MSG( \
        cond,                                 \
        "",                                   \
        __func__,                             \
        ", ",                                 \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", ",                                 \
        ##__VA_ARGS__));                      \
  }

#endif // STRIP_ERROR_MESSAGES
#endif // TORCH_STANDALONE
