#ifdef SD_CUDA
#include <cuda_runtime.h>

#include <graph/profiling/OpTimingTracker.h>

namespace sd {
namespace graph {

// CUDA Event Timer implementation
CudaEventTimer::CudaEventTimer() {
  cudaEvent_t start, stop;

  cudaError_t err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    THROW_EXCEPTION(cudaGetErrorString(err));
  }

  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    THROW_EXCEPTION(cudaGetErrorString(err));
  }

  _startEvent = start;
  _stopEvent = stop;
}

CudaEventTimer::~CudaEventTimer() {
  cudaEventDestroy(static_cast<cudaEvent_t>(_startEvent));
  cudaEventDestroy(static_cast<cudaEvent_t>(_stopEvent));
}

void CudaEventTimer::start() {
  cudaEventRecord(static_cast<cudaEvent_t>(_startEvent));
  _started = true;
}

void CudaEventTimer::stop() {
  if (_started) {
    cudaEventRecord(static_cast<cudaEvent_t>(_stopEvent));
    cudaEventSynchronize(static_cast<cudaEvent_t>(_stopEvent));
  }
}

float CudaEventTimer::elapsedMillis() {
  if (!_started) return -1.0f;
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(_startEvent), static_cast<cudaEvent_t>(_stopEvent));
  return ms;
}

}  // namespace graph
}  // namespace sd
#endif
