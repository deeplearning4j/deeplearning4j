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

#ifndef LIBND4J_RESOURCE_BINDER_H
#define LIBND4J_RESOURCE_BINDER_H

#include <array/NDArray.h>
#include <system/common.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations
class SlotArray;
class PlanTopology;

/**
 * StreamHandle — opaque per-segment stream reference.
 * On CUDA: wraps cudaStream_t.
 * On CPU: nullptr (no-op).
 */
struct StreamHandle {
  void* stream = nullptr;
  int deviceId = -1;
};

/**
 * StreamGuard — RAII stream binding.
 * Sets tl_dspExecutionStream on construction, restores on destruction.
 * On CPU builds: no-op.
 */
class SD_LIB_EXPORT StreamGuard {
 public:
  StreamGuard() : prevStream_(nullptr), active_(false) {}
  explicit StreamGuard(void* newStream);
  ~StreamGuard();

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&& o) noexcept;
  StreamGuard& operator=(StreamGuard&& o) noexcept;

  void* stream() const { return active_ ? currentStream_ : nullptr; }

 private:
  void* prevStream_;
  void* currentStream_ = nullptr;
  bool active_;
};

/**
 * ArgTableHandle — per-segment device argument table.
 * On CUDA: device memory holding kernel argument pointers.
 * On CPU: host memory (or nullptr if no arg table needed).
 */
struct ArgTableHandle {
  void* devicePtr = nullptr;
  size_t sizeBytes = 0;
  int segmentIdx = -1;
};

/**
 * StagingBuffer — pinned host memory for variable external input D2D copies.
 * On CUDA: cudaMallocHost'd buffer used as H2D source for graph arg updates.
 * On CPU: regular heap allocation (no special handling needed).
 */
struct StagingBuffer {
  void* hostPtr = nullptr;
  size_t sizeBytes = 0;
  int externalInputIdx = -1;
  bool inUse = false;
};

/**
 * ResourceBinder — Single RAII owner of all GPU/execution resources for a plan.
 *
 * Replaces scattered resource tracking across:
 *   - TLS stream management (tl_dspExecutionStream set/restore in 6+ places)
 *   - captureWorkspace_ in ExecutionState
 *   - placeholderStagingBuffers_ in NativeDynamicShapePlan
 *   - executionCompleteEvent_ management
 *   - Manual cudaFree calls in 10+ locations
 *
 * RAII guarantee: construction allocates, destruction frees.
 * No other code calls cudaFree/cudaStreamDestroy on plan resources.
 *
 * On CPU builds: the class exists for API uniformity but most methods are no-ops.
 * streamForSegment() returns {nullptr, -1}, captureWorkspace() returns {nullptr, 0},
 * argTable() returns empty handle, syncStagingBuffers() is no-op.
 */
class SD_LIB_EXPORT ResourceBinder {
 public:
  /**
   * Construct ResourceBinder for a plan with the given segment count.
   *
   * @param numSegments   Number of segments in the plan
   * @param numExternalInputs Number of external inputs (for staging buffer sizing)
   * @param deviceId      GPU device ID (-1 for CPU)
   */
  ResourceBinder(int numSegments, int numExternalInputs, int deviceId = -1);
  ~ResourceBinder();

  // Non-copyable, non-movable (owns GPU resources)
  ResourceBinder(const ResourceBinder&) = delete;
  ResourceBinder& operator=(const ResourceBinder&) = delete;
  ResourceBinder(ResourceBinder&&) = delete;
  ResourceBinder& operator=(ResourceBinder&&) = delete;

  // ── Stream management ──────────────────────────────────────────────────

  /**
   * Get the execution stream for a segment.
   * On CUDA: returns a dedicated stream (created lazily on first call).
   * On CPU: returns {nullptr, -1}.
   */
  StreamHandle streamForSegment(int segIdx);

  /**
   * Bind the segment's stream as the active DSP execution stream (TLS).
   * Returns a StreamGuard that restores the previous stream on destruction.
   * On CPU: returns a no-op guard.
   */
  StreamGuard bindStream(int segIdx);

  // ── Capture workspace ──────────────────────────────────────────────────

  /**
   * Get or allocate the capture workspace for a segment.
   * Used during CUDA graph capture to provide temporary memory.
   *
   * @param segIdx  Segment index
   * @param minBytes Minimum workspace size required
   * @return Pointer to workspace memory and its size
   */
  struct WorkspaceHandle {
    void* ptr = nullptr;
    size_t sizeBytes = 0;
  };
  WorkspaceHandle captureWorkspace(int segIdx, size_t minBytes = 0);

  /**
   * Release a segment's capture workspace (e.g., after executor replacement).
   */
  void releaseCaptureWorkspace(int segIdx);

  // ── Argument tables ────────────────────────────────────────────────────

  /**
   * Get or allocate the argument table for a segment.
   * On CUDA: device memory sized to hold all kernel arg pointers.
   * On CPU: returns empty handle.
   *
   * @param segIdx Segment index
   * @param requiredSize Minimum arg table size in bytes
   */
  ArgTableHandle argTable(int segIdx, size_t requiredSize = 0);

  /**
   * Refresh the argument table for a segment using current slot data.
   * Called when address-key comparison detects pointer drift.
   *
   * @param segIdx    Segment index
   * @param inputs    External input array pointers
   * @param numInputs Number of external inputs
   * @param slots     SlotArray with current output pointers
   */
  void refreshArgTable(int segIdx, NDArray** inputs, int numInputs, const SlotArray* slots);

  // ── Staging buffers ────────────────────────────────────────────────────

  /**
   * Sync staging buffers for variable external inputs.
   * Copies changed input data to pinned host memory, then issues D2D.
   * No-op if all inputs have stable pointers (argTableStable equivalent).
   *
   * @param inputs     External input NDArrays
   * @param numInputs  Number of external inputs
   * @param isVariable Flags per input (true = may change between executions)
   * @param stream     Execution stream for async copies
   */
  void syncStagingBuffers(NDArray** inputs, int numInputs,
                          const bool* isVariable, void* stream);

  // ── Cross-stream synchronization ──────────────────────────────────────

  /**
   * Record a completion event on the given stream.
   * Other streams can wait on this event for cross-segment ordering.
   */
  void recordCompletionEvent(void* stream);

  /**
   * Wait for the completion event on the given stream.
   * Ensures cross-segment data dependencies are satisfied.
   */
  void waitForCompletion(void* stream);

  // ── Bulk operations ────────────────────────────────────────────────────

  /**
   * Release all GPU intermediates (streams, workspaces, arg tables, staging).
   * Keeps the ResourceBinder alive for re-warm. Called between
   * prefill/decode transitions.
   *
   * @return Number of resources freed
   */
  int releaseAll();

  /**
   * Release resources for a single segment (e.g., executor replacement).
   */
  void releaseSegment(int segIdx);

  // ── Diagnostics ────────────────────────────────────────────────────────

  /**
   * Total GPU memory held by this ResourceBinder.
   */
  size_t totalAllocatedBytes() const;

  /**
   * Emit resource inventory via DSP_DIAG_MEMORY.
   */
  void emitReport() const;

  // ── Accessors ──────────────────────────────────────────────────────────
  int numSegments() const { return numSegments_; }
  int deviceId() const { return deviceId_; }

 private:
  int numSegments_;
  int numExternalInputs_;
  int deviceId_;

  // Per-segment streams (lazily allocated)
  std::vector<void*> segmentStreams_;

  // Per-segment capture workspaces
  std::vector<void*> captureWorkspaces_;
  std::vector<size_t> captureWorkspaceSizes_;

  // Per-segment argument tables
  std::vector<ArgTableHandle> argTables_;

  // Staging buffers for variable external inputs
  std::vector<StagingBuffer> stagingBuffers_;

  // Cross-stream completion event
  void* completionEvent_ = nullptr;

  // Total bytes tracked for diagnostics
  size_t totalAllocated_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_RESOURCE_BINDER_H
