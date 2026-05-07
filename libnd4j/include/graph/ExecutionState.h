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

#ifndef LIBND4J_EXECUTION_STATE_H
#define LIBND4J_EXECUTION_STATE_H

#include <array/NDArray.h>
#include <graph/SlotBufferOwnership.h>
#include <system/common.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>

namespace sd {
namespace graph {

// Forward declarations
class PlanDefinition;
struct GraphSegment;

/**
 * Per-segment device/stream binding state.
 * The main mutable execution state now lives in GraphSegment::ExecState
 * (NativeDynamicShapePlan.h). This struct retains device/stream binding
 * for ExecutionState's segment management.
 */
struct SegmentExecState {
  int deviceId = -1;               // Which GPU this segment runs on

#ifdef SD_CUDA
  void* stream = nullptr;          // Per-segment CUDA stream (NOT thread-local)
#endif

  void reset() {
    deviceId = -1;
#ifdef SD_CUDA
    stream = nullptr;
#endif
  }
};

/**
 * ExecutionState — Per-plan-instance mutable state, single-thread-bound.
 *
 * One ExecutionState exists per NativeDynamicShapePlan instance.
 * The plan is bound to one thread (error if called from another).
 *
 * Contains:
 *   - slotArrays[totalSlots]: THE single array collection
 *   - ownership[totalSlots]: ownership drives ALL cleanup decisions
 *   - Per-segment device/stream state
 *   - Capture workspace (bump allocator for CUDA graph capture temps)
 *   - Protected weight buffer set
 *
 * Lifecycle state (phase, executeCount) lives in NativeDynamicShapePlan
 * via PlanLifecycle. This class owns per-instance resources only.
 */
class ExecutionState {
 public:
  explicit ExecutionState(int totalOutputSlots);
  ~ExecutionState();

  // Non-copyable, non-movable
  ExecutionState(const ExecutionState&) = delete;
  ExecutionState& operator=(const ExecutionState&) = delete;

  // ── Thread binding ────────────────────────────────────────────────────

  /**
   * Bind this execution state to the calling thread.
   * Called on first execute(). Throws if called from a different thread
   * than the one that first called bind().
   */
  void bindToCurrentThread();

  /**
   * Check that the calling thread is the bound thread.
   * Throws std::runtime_error if called from a different thread.
   */
  void assertBoundThread() const;

  /**
   * Returns true if the state is bound to any thread.
   */
  bool isBound() const { return bound_.load(std::memory_order_relaxed); }

  // ── Slot array access ─────────────────────────────────────────────────

  int totalOutputSlots() const { return totalOutputSlots_; }

  /**
   * Get the slot arrays pointer. This is the SINGLE source of truth
   * for all slot output arrays.
   */
  NDArray** slotArrays() { return slotArrays_; }
  NDArray* slotArray(int idx) const { return slotArrays_[idx]; }
  void setSlotArray(int idx, NDArray* arr) { slotArrays_[idx] = arr; }

  /**
   * Get the ownership array. Drives ALL cleanup decisions.
   */
  SlotBufferInfo* ownership() { return ownership_; }
  const SlotBufferInfo& ownershipAt(int idx) const { return ownership_[idx]; }

  // ── Protected weights ─────────────────────────────────────────────────

  std::unordered_set<DataBuffer*>& protectedWeightBuffers() { return protectedWeightBuffers_; }
  bool isProtectedWeight(DataBuffer* db) const { return protectedWeightBuffers_.count(db) > 0; }

  // ── Capture workspace ────────────────────────────────────────────────

  /**
   * Configure the capture workspace buffer.
   * On GPU: pre-allocated GPU memory for CUDA graph capture temps.
   * On CPU: no-op.
   */
  void setCaptureWorkspace(void* buffer, size_t size);

  /**
   * Bump-allocate from the capture workspace.
   * Returns nullptr if not enough space or no workspace configured.
   * Thread-safe only because ExecutionState is single-thread-bound.
   *
   * @param bytes  Number of bytes to allocate
   * @param align  Alignment (default 256 for GPU buffers)
   * @return Pointer to allocated memory, or nullptr
   */
  void* workspaceAlloc(size_t bytes, size_t align = 256);

  /**
   * Reset the workspace offset to zero (reuse workspace memory).
   */
  void resetWorkspaceOffset() { captureWorkspaceOffset_ = 0; }

  /**
   * Check if capture workspace is configured.
   */
  bool hasWorkspace() const { return captureWorkspace_ != nullptr && captureWorkspaceSize_ > 0; }

  // ── Segment stream/device management ────────────────────────────────

  /**
   * Get the number of segment states.
   */
  int numSegments() const { return numSegments_; }

  /**
   * Allocate segment state array. Called after plan compilation determines
   * segment count.
   */
  void initSegmentStates(int numSegments);

  /**
   * Get segment execution state by index.
   */
  SegmentExecState* segmentState(int segIdx) { return &segmentStates_[segIdx]; }
  const SegmentExecState& segmentStateAt(int segIdx) const { return segmentStates_[segIdx]; }

  /**
   * Bind the device and stream for a segment.
   * On GPU: calls cudaSetDevice and sets tl_dspExecutionStream via RAII.
   * On CPU: no-op.
   *
   * @param segIdx  Segment index
   * @return Previous stream pointer (for RAII restore). Caller must call
   *         restoreSegmentContext() when done.
   */
  void* bindSegmentDevice(int segIdx);

  /**
   * Restore device context after segment execution.
   * @param previousStream  The value returned by bindSegmentDevice()
   * @param previousDevice  Previous CUDA device ID (-1 = don't restore)
   */
  void restoreSegmentContext(void* previousStream, int previousDevice);

  // ── Platform dispatch ─────────────────────────────────────────────────

  /**
   * Free a slot's GPU/CPU memory based on platform.
   * GPU: cudaFreeAsync on the segment's stream
   * CPU: delete the NDArray
   */
  void freeSlotMemory(int slotIdx, void* stream);

  /**
   * Trim GPU memory pool if needed. No-op on CPU.
   */
  void trimPoolIfNeeded();

 private:
  int totalOutputSlots_;

  // THE single array collection — replaces outputSlots + slotArrayCache +
  // pendingClose + deferredClose
  NDArray** slotArrays_;

  // Ownership drives ALL cleanup decisions
  SlotBufferInfo* ownership_;

  // Protected weight DataBuffers (never freed by plan)
  std::unordered_set<DataBuffer*> protectedWeightBuffers_;

  // Per-segment execution state
  SegmentExecState* segmentStates_ = nullptr;
  int numSegments_ = 0;

  // Capture workspace (pre-allocated buffer for CUDA graph capture temps)
  void* captureWorkspace_ = nullptr;
  size_t captureWorkspaceSize_ = 0;
  size_t captureWorkspaceOffset_ = 0;

  // Thread binding
  std::atomic<bool> bound_{false};
  std::thread::id ownerThread_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_EXECUTION_STATE_H
