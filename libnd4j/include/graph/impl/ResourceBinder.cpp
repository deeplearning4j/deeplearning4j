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

#include <graph/ResourceBinder.h>
#include <graph/SlotArray.h>
#include <graph/DspDiagnostics.h>

#include <cstring>
#include <cassert>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#include <array/DataBuffer.h>  // for tl_dspExecutionStream (declared as cudaStream_t)
#endif

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// StreamGuard
// ═══════════════════════════════════════════════════════════════════════════════

StreamGuard::StreamGuard(void* newStream) : prevStream_(nullptr), currentStream_(newStream), active_(true) {
#ifdef SD_CUDA
  prevStream_ = static_cast<void*>(tl_dspExecutionStream);
  tl_dspExecutionStream = static_cast<cudaStream_t>(newStream);
#endif
}

StreamGuard::~StreamGuard() {
  if (active_) {
#ifdef SD_CUDA
    tl_dspExecutionStream = static_cast<cudaStream_t>(prevStream_);
#endif
  }
}

StreamGuard::StreamGuard(StreamGuard&& o) noexcept
    : prevStream_(o.prevStream_), currentStream_(o.currentStream_), active_(o.active_) {
  o.active_ = false;
}

StreamGuard& StreamGuard::operator=(StreamGuard&& o) noexcept {
  if (this != &o) {
    if (active_) {
#ifdef SD_CUDA
      tl_dspExecutionStream = static_cast<cudaStream_t>(prevStream_);
#endif
    }
    prevStream_ = o.prevStream_;
    currentStream_ = o.currentStream_;
    active_ = o.active_;
    o.active_ = false;
  }
  return *this;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ResourceBinder
// ═══════════════════════════════════════════════════════════════════════════════

ResourceBinder::ResourceBinder(int numSegments, int numExternalInputs, int deviceId)
    : numSegments_(numSegments),
      numExternalInputs_(numExternalInputs),
      deviceId_(deviceId) {
  segmentStreams_.resize(numSegments, nullptr);
  captureWorkspaces_.resize(numSegments, nullptr);
  captureWorkspaceSizes_.resize(numSegments, 0);
  argTables_.resize(numSegments);
  stagingBuffers_.resize(numExternalInputs);

#ifdef SD_CUDA
  if (deviceId >= 0) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    completionEvent_ = reinterpret_cast<void*>(event);
  }
#endif

  DSP_DIAG(LIFECYCLE, "ResourceBinder created: %d segments, %d extInputs, device=%d",
           numSegments, numExternalInputs, deviceId);
}

ResourceBinder::~ResourceBinder() {
  releaseAll();

#ifdef SD_CUDA
  if (completionEvent_ != nullptr) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(completionEvent_));
    completionEvent_ = nullptr;
  }
#endif
}

// ── Stream management ──────────────────────────────────────────────────────

StreamHandle ResourceBinder::streamForSegment(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);
  StreamHandle handle;
  handle.deviceId = deviceId_;

#ifdef SD_CUDA
  if (deviceId_ >= 0) {
    if (segmentStreams_[segIdx] == nullptr) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      segmentStreams_[segIdx] = reinterpret_cast<void*>(stream);
      totalAllocated_ += 256;  // Approximate stream overhead
    }
    handle.stream = segmentStreams_[segIdx];
  }
#endif

  return handle;
}

StreamGuard ResourceBinder::bindStream(int segIdx) {
  auto handle = streamForSegment(segIdx);
  return StreamGuard(handle.stream);
}

// ── Capture workspace ──────────────────────────────────────────────────────

ResourceBinder::WorkspaceHandle ResourceBinder::captureWorkspace(int segIdx, size_t minBytes) {
  assert(segIdx >= 0 && segIdx < numSegments_);
  WorkspaceHandle ws;

#ifdef SD_CUDA
  if (deviceId_ >= 0 && minBytes > 0) {
    if (captureWorkspaceSizes_[segIdx] < minBytes) {
      // Free old workspace if exists
      if (captureWorkspaces_[segIdx] != nullptr) {
        totalAllocated_ -= captureWorkspaceSizes_[segIdx];
        cudaFree(captureWorkspaces_[segIdx]);
      }
      // Allocate new
      void* ptr = nullptr;
      cudaMalloc(&ptr, minBytes);
      captureWorkspaces_[segIdx] = ptr;
      captureWorkspaceSizes_[segIdx] = minBytes;
      totalAllocated_ += minBytes;
    }
    ws.ptr = captureWorkspaces_[segIdx];
    ws.sizeBytes = captureWorkspaceSizes_[segIdx];
  }
#endif

  return ws;
}

void ResourceBinder::releaseCaptureWorkspace(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);

#ifdef SD_CUDA
  if (captureWorkspaces_[segIdx] != nullptr) {
    totalAllocated_ -= captureWorkspaceSizes_[segIdx];
    cudaFree(captureWorkspaces_[segIdx]);
    captureWorkspaces_[segIdx] = nullptr;
    captureWorkspaceSizes_[segIdx] = 0;
  }
#endif
}

// ── Argument tables ────────────────────────────────────────────────────────

ArgTableHandle ResourceBinder::argTable(int segIdx, size_t requiredSize) {
  assert(segIdx >= 0 && segIdx < numSegments_);

#ifdef SD_CUDA
  if (deviceId_ >= 0 && requiredSize > 0) {
    auto& table = argTables_[segIdx];
    if (table.sizeBytes < requiredSize) {
      // Free old
      if (table.devicePtr != nullptr) {
        totalAllocated_ -= table.sizeBytes;
        cudaFree(table.devicePtr);
      }
      // Allocate new
      void* ptr = nullptr;
      cudaMalloc(&ptr, requiredSize);
      table.devicePtr = ptr;
      table.sizeBytes = requiredSize;
      table.segmentIdx = segIdx;
      totalAllocated_ += requiredSize;
    }
    return table;
  }
#endif

  return argTables_[segIdx];
}

void ResourceBinder::refreshArgTable(int segIdx, NDArray** inputs, int numInputs,
                                     const SlotArray* slots) {
  assert(segIdx >= 0 && segIdx < numSegments_);

  // The actual arg table refresh is backend-specific.
  // This method provides the entry point — the concrete refresh logic
  // is called from SegmentExecutor which has access to the topology wiring.
  // Here we just mark that a refresh was requested for diagnostics.
  DSP_DIAG(EXECUTE, "arg_table_refresh seg=%d reason=addr_drift", segIdx);
}

// ── Staging buffers ────────────────────────────────────────────────────────

void ResourceBinder::syncStagingBuffers(NDArray** inputs, int numInputs,
                                        const bool* isVariable, void* stream) {
#ifdef SD_CUDA
  if (deviceId_ < 0) return;

  for (int i = 0; i < numInputs && i < numExternalInputs_; i++) {
    if (!isVariable[i]) continue;
    if (inputs[i] == nullptr) continue;

    auto* arr = inputs[i];
    size_t bytes = arr->lengthOf() * arr->sizeOfT();
    if (bytes == 0) continue;

    auto& staging = stagingBuffers_[i];

    // Grow staging buffer if needed
    if (staging.sizeBytes < bytes) {
      if (staging.hostPtr != nullptr) {
        totalAllocated_ -= staging.sizeBytes;
        cudaFreeHost(staging.hostPtr);
      }
      cudaMallocHost(&staging.hostPtr, bytes);
      staging.sizeBytes = bytes;
      staging.externalInputIdx = i;
      totalAllocated_ += bytes;
    }

    // Copy host data to pinned staging, then D2D to device
    void* srcSpecial = arr->specialBuffer();
    if (srcSpecial != nullptr && staging.hostPtr != nullptr) {
      cudaMemcpyAsync(staging.hostPtr, srcSpecial, bytes,
                      cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
    }
    staging.inUse = true;
  }
#endif
}

// ── Cross-stream synchronization ──────────────────────────────────────────

void ResourceBinder::recordCompletionEvent(void* stream) {
#ifdef SD_CUDA
  if (completionEvent_ != nullptr && stream != nullptr) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(completionEvent_),
                    static_cast<cudaStream_t>(stream));
  }
#endif
}

void ResourceBinder::waitForCompletion(void* stream) {
#ifdef SD_CUDA
  if (completionEvent_ != nullptr && stream != nullptr) {
    cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),
                        reinterpret_cast<cudaEvent_t>(completionEvent_), 0);
  }
#endif
}

// ── Bulk operations ────────────────────────────────────────────────────────

int ResourceBinder::releaseAll() {
  int freed = 0;

#ifdef SD_CUDA
  for (int i = 0; i < numSegments_; i++) {
    if (segmentStreams_[i] != nullptr) {
      cudaStreamDestroy(static_cast<cudaStream_t>(segmentStreams_[i]));
      segmentStreams_[i] = nullptr;
      freed++;
    }
    if (captureWorkspaces_[i] != nullptr) {
      cudaFree(captureWorkspaces_[i]);
      captureWorkspaces_[i] = nullptr;
      captureWorkspaceSizes_[i] = 0;
      freed++;
    }
    if (argTables_[i].devicePtr != nullptr) {
      cudaFree(argTables_[i].devicePtr);
      argTables_[i].devicePtr = nullptr;
      argTables_[i].sizeBytes = 0;
      freed++;
    }
  }

  for (int i = 0; i < numExternalInputs_; i++) {
    if (stagingBuffers_[i].hostPtr != nullptr) {
      cudaFreeHost(stagingBuffers_[i].hostPtr);
      stagingBuffers_[i].hostPtr = nullptr;
      stagingBuffers_[i].sizeBytes = 0;
      stagingBuffers_[i].inUse = false;
      freed++;
    }
  }
#endif

  totalAllocated_ = 0;

  DSP_DIAG(MEMORY, "ResourceBinder: releaseAll freed=%d", freed);
  return freed;
}

void ResourceBinder::releaseSegment(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);

#ifdef SD_CUDA
  if (segmentStreams_[segIdx] != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(segmentStreams_[segIdx]));
    segmentStreams_[segIdx] = nullptr;
  }
  releaseCaptureWorkspace(segIdx);
  if (argTables_[segIdx].devicePtr != nullptr) {
    totalAllocated_ -= argTables_[segIdx].sizeBytes;
    cudaFree(argTables_[segIdx].devicePtr);
    argTables_[segIdx].devicePtr = nullptr;
    argTables_[segIdx].sizeBytes = 0;
  }
#endif

  DSP_DIAG(MEMORY, "ResourceBinder: releaseSegment seg=%d", segIdx);
}

// ── Diagnostics ──────────────────────────────────────────────────────────

size_t ResourceBinder::totalAllocatedBytes() const {
  return totalAllocated_;
}

void ResourceBinder::emitReport() const {
  int streamCount = 0;
  size_t wsTotal = 0;
  size_t argTotal = 0;
  size_t stagingTotal = 0;

  for (int i = 0; i < numSegments_; i++) {
    if (segmentStreams_[i] != nullptr) streamCount++;
    wsTotal += captureWorkspaceSizes_[i];
    argTotal += argTables_[i].sizeBytes;
  }
  for (int i = 0; i < numExternalInputs_; i++) {
    stagingTotal += stagingBuffers_[i].sizeBytes;
  }

  DSP_DIAG(MEMORY, "ResourceBinder: streams=%d captureWS=%zuB argTables=%zuB staging=%zuB total=%zuB",
           streamCount, wsTotal, argTotal, stagingTotal, totalAllocated_);
}

}  // namespace graph
}  // namespace sd
