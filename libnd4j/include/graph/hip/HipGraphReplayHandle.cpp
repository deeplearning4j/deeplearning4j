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

#if defined(SD_HIP)

#include <graph/hip/HipGraphReplayHandle.h>
#include <graph/DspDiagnostics.h>
#include <hip/hip_runtime.h>

#include <chrono>

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════════════

HipGraphReplayHandle::HipGraphReplayHandle(int deviceId)
    : deviceId_(deviceId) {
}

HipGraphReplayHandle::~HipGraphReplayHandle() {
  // Release workspace (raw hipFree — no registry available at destruction time).
  // Pool-aware callers should call releaseWorkspace(registry, segIdx) explicitly
  // before destroying the handle. This is the safety net for raw allocations.
  if (captureWorkspacePtr_ != nullptr) {
    hipFree(captureWorkspacePtr_);
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  }
  // Free pinned host pointers
  freeHostPointers();
  // Clean up HIP graph objects
  cleanup();
}

void HipGraphReplayHandle::cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (graphExec_ != nullptr) {
    hipGraphExecDestroy(graphExec_);
    graphExec_ = nullptr;
  }
  if (graph_ != nullptr) {
    hipGraphDestroy(graph_);
    graph_ = nullptr;
  }
  state_ = ReplayState::EMPTY;
  numNodes_ = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Capture lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

bool HipGraphReplayHandle::beginCapture(void* stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clean up any previous graph
  if (graphExec_ != nullptr) {
    hipGraphExecDestroy(graphExec_);
    graphExec_ = nullptr;
  }
  if (graph_ != nullptr) {
    hipGraphDestroy(graph_);
    graph_ = nullptr;
  }

  hipStream_t hipStr = (stream != nullptr) ? *static_cast<hipStream_t*>(stream) : nullptr;

  hipError_t err = hipStreamBeginCapture(hipStr, hipStreamCaptureModeRelaxed);
  if (err != hipSuccess) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: beginCapture failed: %s", hipGetErrorString(err));
    hipGetLastError();  // Clear error state
    state_ = ReplayState::ERRORED;
    return false;
  }

  state_ = ReplayState::CAPTURING;
  captureTimeMs_ = 0.0;

  // Record capture start time
  auto now = std::chrono::high_resolution_clock::now();
  captureTimeMs_ = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();

  DSP_DIAG(EXECUTE, "HipGraphReplayHandle: capture started on device %d", deviceId_);
  return true;
}

bool HipGraphReplayHandle::endCapture(void* stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: endCapture called but not capturing (state=%d)",
             static_cast<int>(state_));
    return false;
  }

  hipStream_t hipStr = (stream != nullptr) ? *static_cast<hipStream_t*>(stream) : nullptr;

  hipError_t err = hipStreamEndCapture(hipStr, &graph_);
  if (err != hipSuccess || graph_ == nullptr) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: endCapture failed: %s",
             hipGetErrorString(err != hipSuccess ? err : hipErrorUnknown));
    hipGetLastError();  // Clear error state
    graph_ = nullptr;
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Record capture end time and compute duration
  auto now = std::chrono::high_resolution_clock::now();
  double endTimeMs = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
  captureTimeMs_ = endTimeMs - captureTimeMs_;

  // Query node count
  size_t count = 0;
  hipGraphGetNodes(graph_, nullptr, &count);
  numNodes_ = count;

  state_ = ReplayState::CAPTURED;
  DSP_DIAG(EXECUTE, "HipGraphReplayHandle: capture ended, %zu nodes, %.2fms", numNodes_, captureTimeMs_);
  return true;
}

bool HipGraphReplayHandle::finalize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ != ReplayState::CAPTURED) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: finalize called but not captured (state=%d)",
             static_cast<int>(state_));
    return false;
  }

  if (graph_ == nullptr) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Destroy any previous exec
  if (graphExec_ != nullptr) {
    hipGraphExecDestroy(graphExec_);
    graphExec_ = nullptr;
  }

  hipError_t err = hipGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
  if (err != hipSuccess || graphExec_ == nullptr) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: instantiate failed: %s",
             hipGetErrorString(err != hipSuccess ? err : hipErrorUnknown));
    hipGetLastError();  // Clear error state
    graphExec_ = nullptr;
    state_ = ReplayState::ERRORED;
    return false;
  }

  state_ = ReplayState::READY;
  DSP_DIAG(EXECUTE, "HipGraphReplayHandle: instantiated (%zu nodes), ready for replay", numNodes_);
  return true;
}

bool HipGraphReplayHandle::replay(void* stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ != ReplayState::READY) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: replay called but not ready (state=%d)",
             static_cast<int>(state_));
    return false;
  }

  if (graphExec_ == nullptr) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  hipStream_t hipStr = (stream != nullptr) ? *static_cast<hipStream_t*>(stream) : nullptr;

  auto startTime = std::chrono::high_resolution_clock::now();

  hipError_t err = hipGraphLaunch(graphExec_, hipStr);
  if (err != hipSuccess) {
    DSP_DIAG(EXECUTE, "HipGraphReplayHandle: launch failed: %s", hipGetErrorString(err));
    hipGetLastError();  // Clear error state
    state_ = ReplayState::ERRORED;
    return false;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  lastReplayTimeMs_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  replayCount_++;

  return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// State queries
// ═══════════════════════════════════════════════════════════════════════════════

ReplayState HipGraphReplayHandle::getState() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

ReplayStatistics HipGraphReplayHandle::getStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);

  ReplayStatistics stats;
  stats.numOperations = static_cast<int>(numNodes_);
  stats.numMemoryOps = 0;  // HIP graph API does not provide per-type node breakdown directly
  stats.estimatedMemory = 0;
  stats.captureTimeMs = captureTimeMs_;
  stats.lastReplayTimeMs = lastReplayTimeMs_;
  stats.replayCount = replayCount_;
  return stats;
}

size_t HipGraphReplayHandle::getNumNodes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (graph_ == nullptr) return 0;

  size_t count = 0;
  hipGraphGetNodes(graph_, nullptr, &count);
  return count;
}

size_t HipGraphReplayHandle::getNumNodesDuringCapture(void* captureStream) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (captureStream == nullptr) return 0;

  hipStream_t hipStr = *static_cast<hipStream_t*>(captureStream);

  // Query the in-progress capture graph via hipStreamGetCaptureInfo
  hipStreamCaptureStatus captureStatus;
  hipGraph_t captureGraph = nullptr;
  hipError_t err = hipStreamGetCaptureInfo(hipStr, &captureStatus, nullptr);
  if (err != hipSuccess || captureStatus != hipStreamCaptureStatusActive) {
    return 0;
  }

  // Use hipStreamGetCaptureInfo_v2 if available to get the graph directly,
  // otherwise fall back to the cached node count
#if HIP_VERSION >= 50300000
  const hipGraphNode_t* deps = nullptr;
  size_t numDeps = 0;
  err = hipStreamGetCaptureInfo_v2(hipStr, &captureStatus, nullptr, &captureGraph, &deps, &numDeps);
  if (err == hipSuccess && captureGraph != nullptr) {
    size_t count = 0;
    hipGraphGetNodes(captureGraph, nullptr, &count);
    return count;
  }
#endif

  // Fallback: return cached count from last endCapture
  return numNodes_;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace management
// ═══════════════════════════════════════════════════════════════════════════════

bool HipGraphReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                              void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  // Note: registryPtr (CaptureBufferRegistry) is CUDA-specific infrastructure.
  // For HIP, we always use raw hipMalloc. If a HIP-specific pool is added later,
  // it can be wired through registryPtr following the same pattern as CUDA.

  hipError_t err = hipMalloc(&captureWorkspacePtr_, bytes);
  if (err == hipSuccess) {
    captureWorkspaceBytes_ = bytes;
    DSP_DIAG(MEMORY, "HipGraphReplayHandle: hipMalloc %zuMB workspace (seg %d, device %d)",
             bytes / (1024 * 1024), segIdx, deviceId);
    return true;
  }

  hipGetLastError();  // Clear error state
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
  DSP_DIAG(MEMORY, "HipGraphReplayHandle: workspace alloc failed (%s)", hipGetErrorString(err));
  return false;
}

void HipGraphReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  hipFree(captureWorkspacePtr_);
  DSP_DIAG(MEMORY, "HipGraphReplayHandle: hipFree workspace (seg %d)", segIdx);

  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void HipGraphReplayHandle::freeHostPointers() {
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) hipHostFree(ptr);
  }
  capturedHostPtrs_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
