#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>

namespace aoti::libtorch_free {
class AOTICudaGuard {
 public:
  AOTICudaGuard(int32_t device_index) : device_(device_index) {
    // Save original device
    cudaGetDevice(&prev_device_);
    // Switch to the target device if necessary
    if (prev_device_ != device_) {
      cudaSetDevice(device_);
    }
  }

  AOTICudaGuard() = delete;
  AOTICudaGuard(const AOTICudaGuard&) = delete;
  AOTICudaGuard& operator=(const AOTICudaGuard&) = delete;
  AOTICudaGuard(AOTICudaGuard&& other) = delete;
  AOTICudaGuard& operator=(AOTICudaGuard&& other) = delete;

  ~AOTICudaGuard() {
    // Restore the original device if necessary
    if (prev_device_ != device_) {
      cudaSetDevice(prev_device_);
    }
  }

  void set_index(int32_t device_index) {
    device_ = device_index;
    cudaSetDevice(device_index);
  }

 private:
  int32_t prev_device_ = 0;
  int32_t device_;
};

static thread_local std::unordered_map<int, cudaStream_t> current_streams;

// Get the current stream for a specific device
inline cudaStream_t get_current_stream(int32_t device) {
  auto it = current_streams.find(device);
  return (it != current_streams.end()) ? it->second : 0; // Default stream is 0
}

// Set the current stream for a specific device
inline void set_current_stream(int32_t device, cudaStream_t stream) {
  current_streams[device] = stream;
}

class AOTICudaStreamGuard {
 public:
  AOTICudaStreamGuard(cudaStream_t stream, int32_t device_index)
      : stream_(stream), device_(device_index) {
    // Save original device
    cudaGetDevice(&prev_device_);
    // Switch to the target device if necessary
    if (prev_device_ != device_) {
      cudaSetDevice(device_);
    }

    // Save the original stream for the current device
    prev_stream_ = get_current_stream(device_);
    // Set the new stream
    set_current_stream(device_, stream_);
  }

  ~AOTICudaStreamGuard() {
    // Restore the original stream for the current device
    set_current_stream(device_, prev_stream_);

    // Restore the original device if necessary
    if (prev_device_ != device_) {
      cudaSetDevice(prev_device_);
    }
  }

  AOTICudaStreamGuard() = delete;
  AOTICudaStreamGuard(const AOTICudaStreamGuard&) = delete;
  AOTICudaStreamGuard& operator=(const AOTICudaStreamGuard&) = delete;
  AOTICudaStreamGuard(AOTICudaStreamGuard&& other) = delete;
  AOTICudaStreamGuard& operator=(AOTICudaStreamGuard&& other) = delete;

 private:
  cudaStream_t prev_stream_; // Original stream on the target device
  cudaStream_t stream_; // Target stream to set
  int32_t prev_device_ = 0; // Original device at construction
  int32_t device_; // Target device for the guard
};
} // namespace aoti::libtorch_free
#endif // USE_CUDA
