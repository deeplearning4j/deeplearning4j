/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifdef SD_CUDA

#include <graph/cuda/CudaGraphReplayHandle.h>
#include <graph/gpu/CaptureBufferRegistry.h>
#include <graph/DspDiagnostics.h>
#include <cuda_runtime.h>

namespace sd {
namespace graph {

CudaGraphReplayHandle::CudaGraphReplayHandle(int deviceId)
    : deviceId_(deviceId),
      handle_(std::make_shared<sd::cuda::CudaGraphHandle>(deviceId)) {
}

CudaGraphReplayHandle::~CudaGraphReplayHandle() {
  // Release workspace (raw cudaFree — no registry available at destruction time).
  // Pool-aware callers should call releaseWorkspace(registry, segIdx) explicitly
  // before destroying the handle. This is the safety net for raw allocations.
  if (captureWorkspacePtr_ != nullptr) {
    cudaFree(captureWorkspacePtr_);
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  }
  // Free pinned host pointers
  freeHostPointers();
  // handle_ shared_ptr cleans up CudaGraphHandle automatically.
  // Capture buffers and external addresses cleaned by base class destructor.
}

bool CudaGraphReplayHandle::beginCapture(void* stream) {
  if (!handle_) return false;
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  return handle_->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
}

bool CudaGraphReplayHandle::endCapture(void* stream) {
  if (!handle_) return false;
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  return handle_->endCapture(cudaStr);
}

bool CudaGraphReplayHandle::finalize() {
  if (!handle_) return false;
  return handle_->instantiate();
}

bool CudaGraphReplayHandle::replay(void* stream) {
  if (!handle_) return false;
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  return handle_->launchAsync(cudaStr);
}

ReplayState CudaGraphReplayHandle::getState() const {
  if (!handle_) return ReplayState::ERROR;
  switch (handle_->getState()) {
    case sd::cuda::GraphState::EMPTY:        return ReplayState::EMPTY;
    case sd::cuda::GraphState::CAPTURING:    return ReplayState::CAPTURING;
    case sd::cuda::GraphState::CAPTURED:     return ReplayState::CAPTURED;
    case sd::cuda::GraphState::INSTANTIATED: return ReplayState::READY;
    case sd::cuda::GraphState::EXECUTING:    return ReplayState::READY;
    case sd::cuda::GraphState::COMPLETED:    return ReplayState::READY;
    case sd::cuda::GraphState::ERROR:        return ReplayState::ERROR;
    default:                                 return ReplayState::ERROR;
  }
}

ReplayStatistics CudaGraphReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  if (!handle_) return stats;

  auto cudaStats = handle_->getStatistics();
  stats.numOperations = cudaStats.numKernels;
  stats.numMemoryOps = cudaStats.numMemcpyH2D + cudaStats.numMemcpyD2H +
                       cudaStats.numMemcpyD2D + cudaStats.numMemsets;
  stats.estimatedMemory = cudaStats.totalMemoryOps;
  stats.captureTimeMs = cudaStats.estimatedTimeMs;
  stats.replayCount = static_cast<int>(handle_->getExecutionTimeline().size());
  return stats;
}

size_t CudaGraphReplayHandle::getNumNodes() const {
  if (!handle_) return 0;
  return handle_->getNumNodes();
}

size_t CudaGraphReplayHandle::getNumNodesDuringCapture(void* captureStream) const {
  if (!handle_) return 0;
  cudaStream_t cudaStr = (captureStream != nullptr)
      ? *static_cast<cudaStream_t*>(captureStream) : nullptr;
  return handle_->getNumNodesDuringCapture(cudaStr);
}

// ── Workspace management (pool-aware) ──────────────────────────────────────

bool CudaGraphReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                               void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  if (registryPtr != nullptr) {
    auto* registry = static_cast<CaptureBufferRegistry*>(registryPtr);
    captureWorkspacePtr_ = registry->allocate(segIdx, bytes, deviceId);
    if (captureWorkspacePtr_ != nullptr) {
      captureWorkspaceBytes_ = bytes;
      DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool-allocated %zuMB workspace (seg %d, device %d)",
               bytes / (1024 * 1024), segIdx, deviceId);
      return true;
    }
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool alloc failed for seg %d, falling back to cudaMalloc",
             segIdx);
  }

  // Fallback: raw cudaMalloc
  cudaError_t err = cudaMalloc(&captureWorkspacePtr_, bytes);
  if (err == cudaSuccess) {
    captureWorkspaceBytes_ = bytes;
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: cudaMalloc %zuMB workspace (seg %d, device %d)",
             bytes / (1024 * 1024), segIdx, deviceId);
    return true;
  }

  cudaGetLastError();  // Clear error
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
  DSP_DIAG(MEMORY, "CudaGraphReplayHandle: workspace alloc failed (%s)", cudaGetErrorString(err));
  return false;
}

void CudaGraphReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  if (registryPtr != nullptr) {
    auto* registry = static_cast<CaptureBufferRegistry*>(registryPtr);
    registry->releaseSegment(segIdx);
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool-released workspace for seg %d", segIdx);
  } else {
    cudaFree(captureWorkspacePtr_);
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: cudaFree workspace");
  }
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void CudaGraphReplayHandle::freeHostPointers() {
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) cudaFreeHost(ptr);
  }
  capturedHostPtrs_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
