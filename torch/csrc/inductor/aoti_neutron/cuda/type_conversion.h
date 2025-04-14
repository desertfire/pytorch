#pragma once
#ifdef USE_CUDA

namespace torch::native::neutron {
void cuda_convertBFloat16ToFloat32(void* src, void* dst, size_t numel);
} // namespace torch::native::neutron
#endif // USE_CUDA
