#pragma once
#ifdef USE_CUDA

namespace torch::native::standalone {
void cuda_convertBFloat16ToFloat32(void* src, void* dst, size_t numel);
} // namespace torch::native::standalone
#endif // USE_CUDA
