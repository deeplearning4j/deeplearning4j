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
#include <graph/ResourceBinderDeviceDispatch.h>

#include <cstring>
#include <cassert>

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// StreamGuard
// ═══════════════════════════════════════════════════════════════════════════════

StreamGuard::StreamGuard(void* newStream) : prevStream_(nullptr), currentStream_(newStream), active_(true) {
  prevStream_ = ResourceBinder_getDspExecutionStream();
  ResourceBinder_setDspExecutionStream(newStream);
}

StreamGuard::~StreamGuard() {
  if (active_) {
    ResourceBinder_setDspExecutionStream(prevStream_);
  }
}

StreamGuard::StreamGuard(StreamGuard&& o) noexcept
    : prevStream_(o.prevStream_), currentStream_(o.currentStream_), active_(o.active_) {
  o.active_ = false;
}

StreamGuard& StreamGuard::operator=(StreamGuard&& o) noexcept {
  if (this != &o) {
    if (active_) {
      ResourceBinder_setDspExecutionStream(prevStream_);
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

  if (deviceId >= 0) {
    completionEvent_ = ResourceBinder_createCompletionEvent();
  }

  DSP_DIAG(LIFECYCLE, "ResourceBinder created: %d segments, %d extInputs, device=%d",
           numSegments, numExternalInputs, deviceId);
}

ResourceBinder::~ResourceBinder() {
  releaseAll();
  ResourceBinder_destroyCompletionEvent(completionEvent_);
  completionEvent_ = nullptr;
}

// ── Stream management ──────────────────────────────────────────────────────

StreamHandle ResourceBinder::streamForSegment(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);
  StreamHandle handle;
  handle.deviceId = deviceId_;

  if (deviceId_ >= 0) {
    if (segmentStreams_[segIdx] == nullptr) {
      segmentStreams_[segIdx] = ResourceBinder_createStream();
      totalAllocated_ += 256;  // Approximate stream overhead
    }
    handle.stream = segmentStreams_[segIdx];
  }

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

  if (deviceId_ >= 0 && minBytes > 0) {
    if (captureWorkspaceSizes_[segIdx] < minBytes) {
      // Free old workspace if exists
      if (captureWorkspaces_[segIdx] != nullptr) {
        totalAllocated_ -= captureWorkspaceSizes_[segIdx];
        ResourceBinder_deviceFree(captureWorkspaces_[segIdx], deviceId_);
      }
      // Allocate new
      captureWorkspaces_[segIdx] = ResourceBinder_deviceAlloc(minBytes, deviceId_);
      captureWorkspaceSizes_[segIdx] = minBytes;
      totalAllocated_ += minBytes;
    }
    ws.ptr = captureWorkspaces_[segIdx];
    ws.sizeBytes = captureWorkspaceSizes_[segIdx];
  }

  return ws;
}

void ResourceBinder::releaseCaptureWorkspace(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);

  if (captureWorkspaces_[segIdx] != nullptr) {
    totalAllocated_ -= captureWorkspaceSizes_[segIdx];
    ResourceBinder_deviceFree(captureWorkspaces_[segIdx], deviceId_);
    captureWorkspaces_[segIdx] = nullptr;
    captureWorkspaceSizes_[segIdx] = 0;
  }
}

// ── Argument tables ────────────────────────────────────────────────────────

ArgTableHandle ResourceBinder::argTable(int segIdx, size_t requiredSize) {
  assert(segIdx >= 0 && segIdx < numSegments_);

  if (deviceId_ >= 0 && requiredSize > 0) {
    auto& table = argTables_[segIdx];
    if (table.sizeBytes < requiredSize) {
      // Free old
      if (table.devicePtr != nullptr) {
        totalAllocated_ -= table.sizeBytes;
        ResourceBinder_deviceFree(table.devicePtr, deviceId_);
      }
      // Allocate new
      table.devicePtr = ResourceBinder_deviceAlloc(requiredSize, deviceId_);
      table.sizeBytes = requiredSize;
      table.segmentIdx = segIdx;
      totalAllocated_ += requiredSize;
    }
    return table;
  }

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
        ResourceBinder_pinnedFree(staging.hostPtr);
      }
      staging.hostPtr = ResourceBinder_pinnedAlloc(bytes);
      staging.sizeBytes = bytes;
      staging.externalInputIdx = i;
      totalAllocated_ += bytes;
    }

    // Copy device data to pinned staging
    void* srcSpecial = arr->specialBuffer();
    if (srcSpecial != nullptr && staging.hostPtr != nullptr) {
      ResourceBinder_memcpyD2HAsync(staging.hostPtr, srcSpecial, bytes, stream);
    }
    staging.inUse = true;
  }
}

// ── Cross-stream synchronization ──────────────────────────────────────────

void ResourceBinder::recordCompletionEvent(void* stream) {
  ResourceBinder_recordEvent(completionEvent_, stream);
}

void ResourceBinder::waitForCompletion(void* stream) {
  ResourceBinder_streamWaitEvent(stream, completionEvent_);
}

// ── Bulk operations ────────────────────────────────────────────────────────

int ResourceBinder::releaseAll() {
  int freed = 0;

  for (int i = 0; i < numSegments_; i++) {
    if (segmentStreams_[i] != nullptr) {
      ResourceBinder_destroyStream(segmentStreams_[i], deviceId_);
      segmentStreams_[i] = nullptr;
      freed++;
    }
    if (captureWorkspaces_[i] != nullptr) {
      ResourceBinder_deviceFree(captureWorkspaces_[i], deviceId_);
      captureWorkspaces_[i] = nullptr;
      captureWorkspaceSizes_[i] = 0;
      freed++;
    }
    if (argTables_[i].devicePtr != nullptr) {
      ResourceBinder_deviceFree(argTables_[i].devicePtr, deviceId_);
      argTables_[i].devicePtr = nullptr;
      argTables_[i].sizeBytes = 0;
      freed++;
    }
  }

  for (int i = 0; i < numExternalInputs_; i++) {
    if (stagingBuffers_[i].hostPtr != nullptr) {
      ResourceBinder_pinnedFree(stagingBuffers_[i].hostPtr);
      stagingBuffers_[i].hostPtr = nullptr;
      stagingBuffers_[i].sizeBytes = 0;
      stagingBuffers_[i].inUse = false;
      freed++;
    }
  }

  totalAllocated_ = 0;

  DSP_DIAG(MEMORY, "ResourceBinder: releaseAll freed=%d", freed);
  return freed;
}

void ResourceBinder::releaseSegment(int segIdx) {
  assert(segIdx >= 0 && segIdx < numSegments_);

  if (segmentStreams_[segIdx] != nullptr) {
    ResourceBinder_destroyStream(segmentStreams_[segIdx], deviceId_);
    segmentStreams_[segIdx] = nullptr;
  }
  releaseCaptureWorkspace(segIdx);
  if (argTables_[segIdx].devicePtr != nullptr) {
    totalAllocated_ -= argTables_[segIdx].sizeBytes;
    ResourceBinder_deviceFree(argTables_[segIdx].devicePtr, deviceId_);
    argTables_[segIdx].devicePtr = nullptr;
    argTables_[segIdx].sizeBytes = 0;
  }

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
