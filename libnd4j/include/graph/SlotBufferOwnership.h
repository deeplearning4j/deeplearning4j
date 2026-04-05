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

#ifndef LIBND4J_SLOT_BUFFER_OWNERSHIP_H
#define LIBND4J_SLOT_BUFFER_OWNERSHIP_H

#include <array/DataBuffer.h>
#include <array/NDArray.h>
#include <system/common.h>

#include <cstdint>

namespace sd {
namespace graph {

/**
 * Describes who owns a slot's output buffer. Replaces ad-hoc tracking
 * (protectedWeightBuffers_, slotViewOutputs_, dedup sets) with a single
 * source of truth per output slot.
 */
enum class BufferOwnership : uint8_t {
  UNSET = 0,          // Not yet determined (first execution)
  SLOT_OWNED,         // Allocated by this slot — freeable on reuse/cleanup
  VIEW_OF_SLOT,       // View into another slot's buffer (parentSlotIdx set)
  VIEW_OF_WEIGHT,     // View into an external weight/constant (never free)
  WEIGHT,             // External constant/variable buffer (never free)
  CAPTURE_BUFFER,     // Owned by GraphReplayHandle capture buffer
  WORKSPACE,          // Workspace-allocated (freed by workspace)
};

/**
 * Per-output-slot ownership metadata. One entry per totalOutputSlots_.
 * Provides O(1) "can I free this?" checks, replacing per-execute HashSet
 * allocation and protectedWeightBuffers_ lookups.
 */
struct SlotBufferInfo {
  BufferOwnership ownership = BufferOwnership::UNSET;
  int parentSlotIdx = -1;       // VIEW_OF_SLOT: which slot's buffer this views
  int deviceId = -1;            // Which GPU this buffer lives on (-1 = CPU/auto)
  DataBuffer* dataBuffer = nullptr;  // Weak ref for identity check (not owned)
  int viewRefCount = 0;              // Number of VIEW_OF_SLOT children referencing this buffer

  /**
   * Returns true if this slot's buffer can be freed during cleanup.
   * Only SLOT_OWNED buffers are freeable — views, weights, and capture
   * buffers are managed by their respective owners.
   */
  SD_INLINE bool canFree() const {
    return ownership == BufferOwnership::SLOT_OWNED;
  }

  /**
   * Returns true if this slot's buffer can be freed RIGHT NOW.
   * Unlike canFree(), this also checks that no VIEW_OF_SLOT children
   * still reference this buffer. This replaces all ad-hoc "is it safe
   * to free?" checks.
   */
  SD_INLINE bool canFreeNow() const {
    return ownership == BufferOwnership::SLOT_OWNED && viewRefCount == 0;
  }

  /**
   * Increment view reference count. Called when a VIEW_OF_SLOT child
   * is created that references this slot's buffer.
   * Asserts that this slot actually owns its buffer (SLOT_OWNED).
   */
  SD_INLINE void addViewRef() {
    assert(ownership == BufferOwnership::SLOT_OWNED &&
           "addViewRef called on non-SLOT_OWNED buffer");
    ++viewRefCount;
  }

  /**
   * Decrement view reference count. Called when a VIEW_OF_SLOT child
   * is freed or no longer references this slot's buffer.
   * Asserts that viewRefCount > 0 before decrement.
   */
  SD_INLINE void removeViewRef() {
    assert(viewRefCount > 0 && "removeViewRef called with viewRefCount == 0");
    --viewRefCount;
  }

  /**
   * Returns true if this slot's output is a view (shares buffer with parent).
   */
  SD_INLINE bool isView() const {
    return ownership == BufferOwnership::VIEW_OF_SLOT ||
           ownership == BufferOwnership::VIEW_OF_WEIGHT;
  }

  /**
   * Returns true if this buffer must never be freed by the plan.
   */
  SD_INLINE bool isProtected() const {
    return ownership == BufferOwnership::WEIGHT ||
           ownership == BufferOwnership::VIEW_OF_WEIGHT ||
           ownership == BufferOwnership::CAPTURE_BUFFER;
  }

  /**
   * Reset to unset state (for reuse after plan invalidation).
   */
  void reset() {
    ownership = BufferOwnership::UNSET;
    parentSlotIdx = -1;
    deviceId = -1;
    dataBuffer = nullptr;
    viewRefCount = 0;
  }
};

/**
 * Utility functions for SlotBufferInfo validation.
 * Implementation in SlotBufferOwnership.cpp.
 */

/**
 * Determine the ownership type for a slot's output by comparing its
 * DataBuffer against external inputs (weights) and other slots.
 *
 * @param outBuffer        DataBuffer of the slot's output
 * @param slotIdx          Index of the slot being classified
 * @param externalInputs   Array of external input NDArrays (weights/placeholders)
 * @param numExternalInputs Number of external inputs
 * @param outputSlots      Array of output NDArrays for all slots
 * @param totalOutputSlots Size of outputSlots array
 * @param parentSlotIdxOut If VIEW_OF_SLOT, receives the parent slot index
 * @return The determined BufferOwnership type
 */
BufferOwnership classifyBufferOwnership(
    DataBuffer* outBuffer,
    int slotIdx,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int* parentSlotIdxOut);

/**
 * Validate ownership array consistency. Returns true if valid.
 * When invalid, logs diagnostic info and returns false.
 *
 * @param ownership       Array of SlotBufferInfo
 * @param totalSlots      Size of ownership array
 * @param outputSlots     Output slot NDArrays for cross-reference
 */
bool validateOwnershipConsistency(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots);

/**
 * Get a human-readable string for a BufferOwnership value.
 */
const char* bufferOwnershipName(BufferOwnership ownership);

/**
 * Classify ownership for a slot that was just executed.
 * This is the single entry point for ownership classification during execution.
 * Called after each slot produces output in executeSlot().
 *
 * @param info           SlotBufferInfo to populate
 * @param outArray       The output NDArray produced by this slot
 * @param slotIdx        Index of the slot being classified
 * @param externalInputs Array of external input NDArrays
 * @param numExternalInputs Number of external inputs
 * @param outputSlots    Array of output NDArrays for all slots
 * @param totalOutputSlots Size of outputSlots array
 * @param ownershipArray Full ownership array for viewRefCount updates
 */
void classifyAndUpdateOwnership(
    SlotBufferInfo& info,
    NDArray* outArray,
    int slotIdx,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    SlotBufferInfo* ownershipArray);

// ═══════════════════════════════════════════════════════════════════════════════
// Phase-Aware Lifecycle Validation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Snapshot of buffer pointers at a phase transition point.
 * Captured when shapes are frozen; compared on every subsequent execute().
 * Pointer drift (address changed) during frozen execution is a hard error.
 */
struct BufferPointerSnapshot {
  int totalSlots = 0;
  void** slotGpuAddresses = nullptr;    // GPU (special) buffer address per slot
  DataBuffer** slotDataBuffers = nullptr; // DataBuffer* identity per slot
  int numExternalInputs = 0;
  void** extGpuAddresses = nullptr;     // GPU address per external input
  DataBuffer** extDataBuffers = nullptr; // DataBuffer* identity per external input
  bool valid = false;                    // True if snapshot has been captured

  ~BufferPointerSnapshot() {
    clear();
  }

  void clear() {
    delete[] slotGpuAddresses; slotGpuAddresses = nullptr;
    delete[] slotDataBuffers; slotDataBuffers = nullptr;
    delete[] extGpuAddresses; extGpuAddresses = nullptr;
    delete[] extDataBuffers; extDataBuffers = nullptr;
    totalSlots = 0;
    numExternalInputs = 0;
    valid = false;
  }

  /**
   * Capture current buffer state as the baseline.
   */
  void capture(NDArray** outputSlots, int numSlots,
               NDArray** externalInputs, int numExt);

  /**
   * Validate that current buffer state matches the snapshot.
   * Returns true if all pointers match. On mismatch, populates errMsg
   * with a description of the first violation found.
   *
   * Checks:
   *   1. Slot GPU addresses unchanged (pointer stability)
   *   2. Slot DataBuffer identities unchanged (no replacement)
   *   3. External input GPU addresses unchanged
   *   4. External input DataBuffers not closed
   *   5. No null slots that were previously non-null (freed)
   */
  bool validate(NDArray** outputSlots, int numSlots,
                NDArray** externalInputs, int numExt,
                char* errMsg, int errMsgLen) const;
};

/**
 * Phase-aware lifecycle validation. Checks that the current state of all
 * output slots and external inputs is consistent with the plan phase.
 *
 * Called at the start of execute() and after each segment.
 * Returns true if valid; on violation, populates errMsg and returns false.
 *
 * Phase-specific checks:
 *   SLOT_BY_SLOT: ownership consistency only
 *   SHAPES_FROZEN: + no closed DataBuffers in live slots
 *   POINTERS_STABLE: + buffer addresses match snapshot
 *   REPLAYING: + replay handles intact, no capture buffer drift
 */
bool validateLifecycleForPhase(
    int planPhase,  // PlanPhase as int (to avoid circular include)
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers,
    const BufferPointerSnapshot* snapshot,
    char* errMsg, int errMsgLen);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SLOT_BUFFER_OWNERSHIP_H
