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
#include <memory/cuda/CudaMemoryPool.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <graph/gpu/CapturedModuleRegistry.h>
#include <execution/LaunchContext.h>

namespace sd {
namespace graph {

CudaGraphReplayHandle::CudaGraphReplayHandle(int deviceId)
    : deviceId_(deviceId),
      handle_(std::make_shared<sd::cuda::CudaGraphHandle>(deviceId)) {
}

CudaGraphReplayHandle::~CudaGraphReplayHandle() {
  // Release workspace via pool.free — no registry available at destruction time.
  // Pool-aware callers should call releaseWorkspace(registry, segIdx) explicitly
  // before destroying the handle. This is the safety net for direct allocations.
  if (captureWorkspacePtr_ != nullptr && !workspaceIsExternal_) {
    // Unregister from CudaMemoryPool so free() no longer skips interior pointers
    memory::CudaMemoryPool::getInstance().unregisterCaptureWorkspace(captureWorkspacePtr_);
    memory::CudaMemoryPool::getInstance().free(captureWorkspacePtr_, deviceId_);
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  } else if (workspaceIsExternal_) {
    // External workspace — just clear our reference, owner frees it
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
  DSP_DIAG_DEV(EXECUTE, deviceId_,
               "CudaGraphReplayHandle::beginCapture stream=%p device=%d hostPtrs=%d",
               (void*)cudaStr, deviceId_, (int)capturedHostPtrs_.size());
  bool ok = handle_->beginCapture(cudaStr, cudaStreamCaptureModeThreadLocal);
  if (!ok) {
    DSP_DIAG_DEV(FALLBACK, deviceId_, "CudaGraphReplayHandle::beginCapture FAILED device=%d", deviceId_);
  }
  return ok;
}

bool CudaGraphReplayHandle::endCapture(void* stream) {
  if (!handle_) return false;
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  bool ok = handle_->endCapture(cudaStr);
  DSP_DIAG_DEV(EXECUTE, deviceId_,
               "CudaGraphReplayHandle::endCapture %s device=%d nodes=%zu",
               ok ? "OK" : "FAILED", deviceId_, ok ? handle_->getNumNodes() : 0);
  return ok;
}

bool CudaGraphReplayHandle::finalize() {
  if (!handle_) return false;
  bool ok = handle_->instantiate();
  DSP_DIAG_DEV(COMPILE, deviceId_,
               "CudaGraphReplayHandle::finalize (instantiate) %s device=%d",
               ok ? "OK" : "FAILED", deviceId_);
  return ok;
}

bool CudaGraphReplayHandle::replay(void* stream) {
  if (!handle_) {
    DSP_DIAG_DEV(FALLBACK, deviceId_,
                 "CudaGraphReplayHandle::replay: handle_ is NULL device=%d", deviceId_);
    auto* errorRef = LaunchContext::defaultContext()->errorReference();
    errorRef->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
    errorRef->setErrorMessage("CUDA graph replay handle is null");
    return false;
  }
  cudaStream_t cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  cudaError_t fastLaunchError = cudaSuccess;

  // Fast path: call cudaGraphLaunch directly, bypassing launchAsync overhead
  // (mutex, cudaGetDevice, chrono timestamps, cudaGetLastError, state transitions,
  // timeline tracking). Safe because:
  //   1. isReady() was already checked at the call site (compositeReplay)
  //   2. Device doesn't change during composite replay (single-device steady state)
  //   3. State transitions are cosmetic (INSTANTIATED/EXECUTING/COMPLETED all map to READY)
  //   4. getGraphExec() is stable after instantiate — no concurrent mutation during replay
  cudaGraphExec_t exec = handle_->getGraphExec();
  if (exec != nullptr) {
    DSP_DIAG_DEV(EXECUTE, deviceId_,
                 "CudaGraphReplayHandle::replay cudaGraphLaunch exec=%p stream=%p device=%d "
                 "nodes=%zu state=%d",
                 (void*)exec, (void*)cudaStr, deviceId_,
                 handle_->getNumNodes(), (int)handle_->getState());
    cudaError_t err = cudaGraphLaunch(exec, cudaStr);
    if (err == cudaSuccess) { replayCount_++; return true; }
    fastLaunchError = err;
    // Launch failed — fall through to full launchAsync for detailed diagnostics
    DSP_DIAG_DEV(FALLBACK, deviceId_,
                 "CudaGraphReplayHandle::replay cudaGraphLaunch FAILED err=%d (%s) exec=%p stream=%p",
                 (int)err, cudaGetErrorString(err), (void*)exec, (void*)cudaStr);
    cudaGetLastError();  // Clear the error before retrying via launchAsync
  } else {
    DSP_DIAG_DEV(FALLBACK, deviceId_,
                 "CudaGraphReplayHandle::replay: graphExec is NULL — using slow path device=%d state=%d",
                 deviceId_, (int)handle_->getState());
  }

  // Slow path: full launchAsync with diagnostics
  bool ok = handle_->launchAsync(cudaStr);
  if (ok) {
    replayCount_++;
  } else {
    DSP_DIAG_DEV(FALLBACK, deviceId_,
                 "CudaGraphReplayHandle::replay launchAsync FAILED device=%d state=%d",
                 deviceId_, (int)handle_->getState());
    char detail[512];
    std::snprintf(detail, sizeof(detail),
                  "CUDA graph replay failed on device %d: fastLaunchError=%d (%s), "
                  "graphExec=%p, stream=%p, state=%d",
                  deviceId_, static_cast<int>(fastLaunchError),
                  fastLaunchError == cudaSuccess ? "not attempted" : cudaGetErrorString(fastLaunchError),
                  static_cast<void*>(exec), static_cast<void*>(cudaStr),
                  static_cast<int>(handle_->getState()));
    auto* errorRef = LaunchContext::defaultContext()->errorReference();
    errorRef->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
    errorRef->setErrorMessage(detail);
  }
  return ok;
}

ReplayState CudaGraphReplayHandle::getState() const {
  if (!handle_) return ReplayState::ERRORED;
  switch (handle_->getState()) {
    case sd::cuda::GraphState::EMPTY:        return ReplayState::EMPTY;
    case sd::cuda::GraphState::CAPTURING:    return ReplayState::CAPTURING;
    case sd::cuda::GraphState::CAPTURED:     return ReplayState::CAPTURED;
    case sd::cuda::GraphState::INSTANTIATED: return ReplayState::READY;
    case sd::cuda::GraphState::EXECUTING:    return ReplayState::READY;
    case sd::cuda::GraphState::COMPLETED:    return ReplayState::READY;
    case sd::cuda::GraphState::ERROR:        return ReplayState::ERRORED;
    default:                                 return ReplayState::ERRORED;
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
  stats.replayCount = replayCount_;
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
      // Register so CudaMemoryPool::free() skips interior pointers from this workspace
      memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(captureWorkspacePtr_, bytes);
      DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool-allocated %zuMB workspace (seg %d, device %d)",
               bytes / (1024 * 1024), segIdx, deviceId);
      return true;
    }
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: registry alloc failed for seg %d, falling back to allocateDirect",
             segIdx);
  }

  // Fallback: allocateDirect routes through CudaMemoryPool on a dedicated
  // non-capturing stream — capture-safe (no graph mem-node), tracked so free()
  // routes to cudaFreeAsync on the same non-capturing stream.
  captureWorkspacePtr_ = memory::CudaMemoryPool::getInstance().allocateDirect(bytes, deviceId);
  if (captureWorkspacePtr_ != nullptr) {
    captureWorkspaceBytes_ = bytes;
    // Register so CudaMemoryPool::free() skips interior pointers from this workspace
    memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(captureWorkspacePtr_, bytes);
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: allocateDirect %zuMB workspace (seg %d, device %d)",
             bytes / (1024 * 1024), segIdx, deviceId);
    return true;
  }

  // allocateDirect returned nullptr — pool OOM.
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
  DSP_DIAG(MEMORY, "CudaGraphReplayHandle: workspace alloc FAILED on device %d — "
           "capture will be skipped for this segment. GPU %d may be out of memory.",
           deviceId, deviceId);
  return false;
}

void CudaGraphReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  if (workspaceIsExternal_) {
    // External workspace — just clear our reference, owner frees it
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
    return;
  }

  // Unregister from CudaMemoryPool before freeing
  memory::CudaMemoryPool::getInstance().unregisterCaptureWorkspace(captureWorkspacePtr_);

  void* wsBasePtr = captureWorkspacePtr_;
  size_t wsBytes  = captureWorkspaceBytes_;
  if (registryPtr != nullptr) {
    auto* registry = static_cast<CaptureBufferRegistry*>(registryPtr);
    registry->releaseSegment(segIdx);
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool-released workspace seg=%d base=%p size=%zu "
             "(any TAD-offset interior ptrs from this workspace are now invalid if not standalone)",
             segIdx, wsBasePtr, wsBytes);
  } else {
    memory::CudaMemoryPool::getInstance().free(captureWorkspacePtr_, deviceId_);
    DSP_DIAG(MEMORY, "CudaGraphReplayHandle: pool.free workspace base=%p size=%zu "
             "(any TAD-offset interior ptrs from this workspace are now invalid if not standalone)",
             wsBasePtr, wsBytes);
  }
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void CudaGraphReplayHandle::freeHostPointers() {
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) cudaFreeHost(ptr);
  }
  capturedHostPtrs_.clear();
  // Captured modules die WITH the graph: only now are the baked kernel nodes
  // gone, so cuModuleUnload is finally legal.
  for (auto* mod : capturedModules_) {
    if (mod == nullptr) continue;
    // The Triton backend is a SINGLETON whose compiled cache outlives plans:
    // this module may still be served to later plans (cache hit) or baked by
    // other handles. Only the LAST holder unloads.
    if (sd::graph::modreg::releaseFromHandle(mod)) {
      DSP_DIAG_DEV(EXECUTE, deviceId_,
                   "CudaGraphReplayHandle: last ref — unloading captured module=%p", mod);
      cuModuleUnload(static_cast<CUmodule>(mod));
    } else {
      DSP_DIAG_DEV(EXECUTE, deviceId_,
                   "CudaGraphReplayHandle: released captured module=%p (other holders remain)", mod);
    }
  }
  capturedModules_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
