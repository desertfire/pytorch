#pragma once

#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <memory>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/backends/backend_interface.h>

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace torch::jit::aoti {

// Each instance owns one single-threaded runner. Calls must not overlap, must
// come from one host thread, and must stay on one ordered current stream unless
// the caller externally synchronizes before switching streams.
class TORCH_API AOTIBackend final : public PyTorchBackendInterface {
 public:
  AOTIBackend() = default;
  ~AOTIBackend() override;

  bool is_available() override;

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override;

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override;

 private:
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};

} // namespace torch::jit::aoti

#endif
