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
#include <unordered_set>

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
           ownership == BufferOwnership::VIEW_OF_WEIGHT;
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
  int* slotDeviceIds = nullptr;          // Device ID where each slot buffer lives
  void** slotPrimaryAddresses = nullptr; // Host (primary) buffer address per slot (may be null)
  LongType** slotShapeInfoAddresses = nullptr; // Device-side shape info pointer per slot
  NDArray** slotNDArrayIdentity = nullptr;     // NDArray wrapper identity per slot
  LongType* slotBufferOffsets = nullptr;       // NDArray::offset() at capture time
  LongType* slotLengths = nullptr;             // NDArray::lengthOf() at capture time
  uint8_t* slotActualityFlags = nullptr;       // bit0 = isPrimaryActual, bit1 = isSpecialActual
  char* slotOrderings = nullptr;               // NDArray::ordering() at capture time ('c' or 'f')
  int numExternalInputs = 0;
  void** extGpuAddresses = nullptr;     // GPU address per external input
  DataBuffer** extDataBuffers = nullptr; // DataBuffer* identity per external input
  int* extDeviceIds = nullptr;           // Device ID where each ext input buffer lives
  void** extPrimaryAddresses = nullptr;  // Host (primary) buffer address per ext input (may be null)
  LongType** extShapeInfoAddresses = nullptr;  // Device-side shape info pointer per ext input
  NDArray** extNDArrayIdentity = nullptr;      // NDArray wrapper identity per ext input
  LongType* extBufferOffsets = nullptr;        // NDArray::offset() at capture time
  LongType* extLengths = nullptr;              // NDArray::lengthOf() at capture time
  uint8_t* extActualityFlags = nullptr;        // bit0 = isPrimaryActual, bit1 = isSpecialActual
  char* extOrderings = nullptr;                // NDArray::ordering() at capture time ('c' or 'f')
  int capturedDeviceId = -1;             // Device ID where the plan was executing at capture time
  bool valid = false;                    // True if snapshot has been captured

  ~BufferPointerSnapshot() {
    clear();
  }

  void clear() {
    delete[] slotGpuAddresses; slotGpuAddresses = nullptr;
    delete[] slotDataBuffers; slotDataBuffers = nullptr;
    delete[] slotDeviceIds; slotDeviceIds = nullptr;
    delete[] slotPrimaryAddresses; slotPrimaryAddresses = nullptr;
    delete[] slotShapeInfoAddresses; slotShapeInfoAddresses = nullptr;
    delete[] slotNDArrayIdentity; slotNDArrayIdentity = nullptr;
    delete[] slotBufferOffsets; slotBufferOffsets = nullptr;
    delete[] slotLengths; slotLengths = nullptr;
    delete[] slotActualityFlags; slotActualityFlags = nullptr;
    delete[] slotOrderings; slotOrderings = nullptr;
    delete[] extGpuAddresses; extGpuAddresses = nullptr;
    delete[] extDataBuffers; extDataBuffers = nullptr;
    delete[] extDeviceIds; extDeviceIds = nullptr;
    delete[] extPrimaryAddresses; extPrimaryAddresses = nullptr;
    delete[] extShapeInfoAddresses; extShapeInfoAddresses = nullptr;
    delete[] extNDArrayIdentity; extNDArrayIdentity = nullptr;
    delete[] extBufferOffsets; extBufferOffsets = nullptr;
    delete[] extLengths; extLengths = nullptr;
    delete[] extActualityFlags; extActualityFlags = nullptr;
    delete[] extOrderings; extOrderings = nullptr;
    totalSlots = 0;
    numExternalInputs = 0;
    capturedDeviceId = -1;
    valid = false;
  }

  /**
   * Capture current buffer state as the baseline.
   * Records all pointer/identity/metadata fields for full frozen-phase tracking.
   */
  void capture(NDArray** outputSlots, int numSlots,
               NDArray** externalInputs, int numExt);

  /**
   * Null out snapshot entries for "transient" slots — VIEW_OF_WEIGHT slots
   * whose underlying DataBuffer is NOT in protectedWeightBuffers.  These
   * slots are views over placeholder external inputs; the placeholder is
   * supplied fresh each call and its old DataBuffer is legitimately closed
   * between calls.  The view wrapper is refreshed by slot-exec on the next
   * call, so the snapshot must not treat the old state as authoritative.
   *
   * Safe to call multiple times (idempotent).  No-op if the snapshot is
   * invalid or ownership is null.
   */
  void pruneTransientViewSlots(
      const SlotBufferInfo* ownership,
      const std::unordered_set<DataBuffer*>& protectedWeightBuffers);

  /**
   * Validate that current buffer state matches the snapshot.
   * THROW_EXCEPTION on any mismatch — no fallback, no slot-by-slot recovery.
   *
   * Checks (for slots AND external inputs):
   *   1. GPU (special) address unchanged (pointer stability)
   *   2. DataBuffer identity unchanged (no replacement)
   *   3. Host (primary) address unchanged (no host reallocation)
   *   4. Device-side shape info pointer unchanged (no re-registration)
   *   5. NDArray wrapper identity unchanged (no wrapper replacement)
   *   6. Buffer offset unchanged (no view base move)
   *   7. lengthOf() unchanged (shape-frozen invariant)
   *   8. Actuality flags unchanged (no unexpected primary/special flips)
   *   9. DataBuffer not closed
   *   10. No null slots that were previously non-null (freed)
   *   11. Device IDs unchanged (no buffer migration)
   *   12. Memory ordering unchanged (no layout flip)
   */
  bool validate(NDArray** outputSlots, int numSlots,
                NDArray** externalInputs, int numExt,
                char* errMsg, int errMsgLen) const;

  /**
   * Detect stale data via actuality flag transitions.
   * From SHAPES_FROZEN onward, output slots that belong to the graph should
   * have sAct=1 (device authoritative). If pAct transitions to 1 without
   * sAct (indicating a host write overwrote device data), that's stale data.
   * THROW_EXCEPTION on any unexpected transition.
   */
  void detectStaleActualityTransitions(NDArray** outputSlots, int numSlots,
                                       NDArray** externalInputs, int numExt,
                                       int planPhase) const;
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
 *   REPLAYING: + buffer addresses match snapshot, replay handles intact, no capture buffer drift
 */
bool validateLifecycleForPhase(
    int planPhase,  // PlanPhase as int (to avoid circular include)
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers,
    const BufferPointerSnapshot* snapshot,
    char* errMsg, int errMsgLen);

/**
 * Validate viewRefCount consistency across all slots.
 * For each SLOT_OWNED slot, verify that viewRefCount matches the actual
 * number of VIEW_OF_SLOT children that reference it.
 * Returns true if consistent; logs violations and returns false otherwise.
 */
bool validateViewRefCounts(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots);

/**
 * Detect stale ownership entries — slots whose tracked DataBuffer no longer
 * matches the actual NDArray's DataBuffer. This indicates a slot was
 * replaced (new allocation or view swap) without re-classifying ownership.
 * Returns the number of stale entries found (0 = all consistent).
 * Logs each stale entry via DSP_DIAG(MEMORY, ...).
 */
int detectStaleOwnership(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots);

/**
 * Detect closed DataBuffers in live slots. Returns count of closed buffers found.
 * Logs each via DSP_DIAG(MEMORY, ...) with slot index and ownership type.
 */
int detectClosedBuffers(
    const SlotBufferInfo* ownership, int totalSlots,
    NDArray** outputSlots,
    NDArray** externalInputs, int numExternalInputs);

// ═══════════════════════════════════════════════════════════════════════════════
// Always-On Array Validity Assertions
//
// These check that an NDArray is in a valid state for execution. Every check
// is a plain field read — no sync, no allocation, no side effects. Designed
// to run at EVERY transition point (pre-slot, post-slot, segment boundary,
// JNI entry) without measurable overhead.
//
// NOT gated behind diagnostics. Invalid arrays are caught immediately with
// a rich diagnostic message identifying exactly WHAT failed, WHERE (slot,
// segment, phase), and WHEN (execution count).
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Identifies exactly which validity check failed for an NDArray.
 * Ordered from most severe (use-after-free) to least (missing GPU pointer).
 */
enum class ArrayInvalidReason : uint8_t {
  VALID = 0,              // All checks passed
  NULL_ARRAY,             // NDArray* is nullptr (unexpected for a wired slot)
  NULL_SHAPE_INFO,        // shapeInfo() is nullptr (corrupted or uninitialized)
  NULL_DATABUFFER,        // dataBuffer() is nullptr on a non-empty array
  DESTROYED_DATABUFFER,   // magic == 0xDEADBEEF: use-after-free
  CLOSED_DATABUFFER,      // isClosed() == true: freed between steps
  INVALID_DATABUFFER,     // isValid() false but not destroyed/closed (corrupt magic)
  NULL_GPU_POINTER,       // special() is nullptr on CUDA for a non-empty array
};

/**
 * Human-readable name for an ArrayInvalidReason value.
 */
const char* arrayInvalidReasonName(ArrayInvalidReason reason);

/**
 * Validate a single NDArray for execution readiness.
 * Returns VALID if the array can be safely passed to a kernel.
 * Returns the first failing check otherwise.
 *
 * NULL arrays return NULL_ARRAY (callers that allow null inputs should
 * check for null before calling this).
 * Empty arrays (isEmpty() == true) always return VALID — they have no
 * DataBuffer by design.
 *
 * Cost: 3-5 field reads (pointer, magic number, closed flag, GPU pointer).
 * No sync, no allocation, no side effects.
 */
ArrayInvalidReason validateArrayForExecution(const NDArray* arr);

/**
 * Validate all inputs for a slot BEFORE execution.
 * Resolves each input from outputSlots (for slot-sourced) or externalInputs
 * (for external-sourced) using inputSourceIndices, then validates each.
 *
 * Returns 0 if all inputs valid, or the count of invalid inputs.
 * On failure, populates errMsg with details about the FIRST invalid input.
 *
 * Always-on — not gated behind diagnostics.
 *
 * @param slotIdx           The slot about to execute
 * @param opName            Op name for diagnostic messages
 * @param outputSlots       Plan's output slot array
 * @param totalOutputSlots  Size of outputSlots
 * @param externalInputs    External input array (weights, placeholders)
 * @param numExternalInputs Size of externalInputs
 * @param inputSourceIndices Wiring: >=0 means outputSlots[idx], <0 means externalInputs[-(idx+1)]
 * @param numInputs         Number of inputs for this slot
 * @param executeCount      Current plan execution count (for diagnostics)
 * @param planPhase         Current plan phase as int (for diagnostics)
 * @param errMsg            Buffer for error message (null-safe)
 * @param errMsgLen         Size of errMsg buffer
 */
int validateSlotInputs(
    int slotIdx, const char* opName,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalInputs, int numExternalInputs,
    const int* inputSourceIndices, int numInputs,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen);

/**
 * Validate all outputs for a slot AFTER execution.
 * Checks each output slot index to verify the produced array is valid.
 *
 * Returns 0 if all outputs valid, or the count of invalid outputs.
 * On failure, populates errMsg with details about the FIRST invalid output.
 *
 * Always-on — not gated behind diagnostics.
 */
int validateSlotOutputs(
    int slotIdx, const char* opName,
    NDArray** outputSlots, int totalOutputSlots,
    const int* outputSlotIndices, int numOutputs,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen);

/**
 * Validate all arrays in a slot range [startSlot, endSlot].
 * Used at segment boundaries to verify all outputs in the completed
 * segment are valid before the next segment reads them as inputs.
 *
 * Returns 0 if all valid, or the count of invalid arrays.
 * On failure, populates errMsg with details about the FIRST invalid array.
 */
int validateSlotRange(
    int startSlot, int endSlot,
    NDArray** outputSlots, int totalOutputSlots,
    int executeCount, int planPhase,
    char* errMsg, int errMsgLen);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SLOT_BUFFER_OWNERSHIP_H
