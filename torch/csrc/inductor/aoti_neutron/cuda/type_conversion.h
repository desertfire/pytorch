#pragma once
#ifdef USE_CUDA

namespace torch::neutron {
void cuda_convertBFloat16ToFloat32(void* src, void* dst, size_t numel);
} // namespace torch::neutron
#endif // USE_CUDA
