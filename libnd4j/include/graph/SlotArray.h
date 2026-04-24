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

#ifndef LIBND4J_SLOT_ARRAY_H
#define LIBND4J_SLOT_ARRAY_H

#include <array/NDArray.h>
#include <graph/SlotBufferOwnership.h>
#include <system/common.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_set>

namespace sd {
namespace graph {

// Forward declaration
class PlanTopology;

// Matches NativeSlot::SlotState — kept here so SlotArray doesn't depend on
// NativeDynamicShapePlan.h (which is being incrementally decomposed).
enum class SlotLifecycleState : uint8_t {
  WARMUP = 0,
  SHAPE_CACHED = 1,
  FROZEN = 2,
  FROZEN_CONSTANT = 3,
};

// Maximum number of untracked outputs per slot (multi-output ops).
// Most ops have 1 output; only a few (split, unstack) have multiple.
static constexpr int SLOT_MAX_UNTRACKED_OUTPUTS = 8;

/**
 * SlotEntry — unified per-slot storage.
 *
 * Combines what was previously scattered across 4+ parallel arrays:
 *   - NDArray* (from outputSlots_)
 *   - SlotBufferInfo (from slotOwnership_)
 *   - bool isViewProducer (from slotIsViewProducer_)
 *   - NDArray* untrackedOutputs (from untrackedOutputCache_)
 *
 * Plus slot-level mutable state that was on NativeSlot (which becomes immutable
 * once moved to PlanTopology):
 *   - SlotLifecycleState (was NativeSlot::state_)
 *   - uint32_t generation (was NativeSlot::generation_)
 */
struct SlotEntry {
  // The output NDArray for this slot (may be null during warmup)
  NDArray* array = nullptr;

  // Ownership classification — drives all cleanup decisions
  SlotBufferInfo ownership;

  // Lifecycle state — mutable per-execution, independent of topology
  SlotLifecycleState state = SlotLifecycleState::WARMUP;

  // View producer flag — set during warmup when the slot creates a view
  bool isViewProducer = false;

  // Write generation counter — bumped on every write to this slot's output
  uint32_t generation = 0;

  // Untracked outputs for multi-output ops (e.g., split produces N outputs,
  // only one goes into the main slot; the rest are here)
  NDArray* untrackedOutputs[SLOT_MAX_UNTRACKED_OUTPUTS] = {};

  // Convenience: check if this slot can be freed
  bool canFree() const { return ownership.canFree(); }
  bool canFreeNow() const { return ownership.canFreeNow(); }

  // Convenience: check state
  bool isFrozen() const { return state >= SlotLifecycleState::FROZEN; }
  bool isFrozenConstant() const { return state == SlotLifecycleState::FROZEN_CONSTANT; }
  bool shapeCacheValid() const { return state >= SlotLifecycleState::SHAPE_CACHED; }

  // Bump generation and return new value
  uint32_t bumpGeneration() { return ++generation; }
};

/**
 * SlotArray — Unified buffer storage for all plan output slots.
 *
 * Single source of truth for:
 *   - Output NDArray pointers (replaces outputSlots_)
 *   - Ownership metadata (replaces slotOwnership_)
 *   - View producer flags (replaces slotIsViewProducer_)
 *   - Untracked outputs (replaces untrackedOutputCache_)
 *   - Slot lifecycle state (replaces NativeSlot::state_)
 *   - Protected weight buffer set (replaces protectedWeightBuffers_)
 *
 * Thread-safety: single-thread-bound. Asserts if accessed from wrong thread.
 *
 * "Who owns what?" → check entries_[idx].ownership
 * "Is this slot frozen?" → check entries_[idx].state
 * One collection, not six.
 */
class SD_LIB_EXPORT SlotArray {
 public:
  explicit SlotArray(int totalSlots);
  ~SlotArray();

  // Non-copyable, non-movable
  SlotArray(const SlotArray&) = delete;
  SlotArray& operator=(const SlotArray&) = delete;
  SlotArray(SlotArray&&) = delete;
  SlotArray& operator=(SlotArray&&) = delete;

  // ── Size ────────────────────────────────────────────────────────────────
  int totalSlots() const { return totalSlots_; }

  // ── Element access ──────────────────────────────────────────────────────
  SlotEntry& operator[](int idx) { return entries_[idx]; }
  const SlotEntry& operator[](int idx) const { return entries_[idx]; }

  // ── Array access (convenience) ──────────────────────────────────────────
  NDArray* array(int idx) const { return entries_[idx].array; }

  /**
   * Raw pointer array for JNI/op dispatch.
   * Returns NDArray** that can be passed to ops expecting a flat array.
   * Kept in sync with entries_[i].array by setArray().
   */
  NDArray** rawPtrArray() { return rawArrays_; }
  NDArray* const* rawPtrArray() const { return rawArrays_; }

  // ── Mutation ────────────────────────────────────────────────────────────

  /**
   * Set an array at the given slot index with ownership classification.
   * This is the ONLY way to install an array into a slot.
   *
   * @param idx           Slot index
   * @param arr           NDArray to install (may be null to clear)
   * @param ownershipType Ownership classification
   * @param parentSlotIdx Parent slot index (for VIEW_OF_SLOT), -1 otherwise
   */
  void setArray(int idx, NDArray* arr, BufferOwnership ownershipType, int parentSlotIdx = -1);

  /**
   * Set slot lifecycle state.
   */
  void setState(int idx, SlotLifecycleState newState);

  /**
   * Mark a slot as a view producer.
   */
  void setViewProducer(int idx, bool isView);

  // ── Protected weights ──────────────────────────────────────────────────

  /**
   * Register a DataBuffer as a protected weight (never freed by plan).
   */
  void addProtectedWeight(DataBuffer* db);

  /**
   * Check if a DataBuffer is a protected weight.
   */
  bool isProtectedWeight(DataBuffer* db) const;

  /**
   * Get the full protected weight set (for plan freeze operations).
   */
  const std::unordered_set<DataBuffer*>& protectedWeights() const { return protectedWeightBuffers_; }

  // ── Bulk operations ─────────────────────────────────────────────────────

  /**
   * Reset all slots to initial state (for releaseGpuIntermediates).
   * If keepWeights is true, preserves WEIGHT-owned slots.
   */
  void resetAll(bool keepWeights);

  /**
   * Demote all FROZEN/FROZEN_CONSTANT slots back to SHAPE_CACHED.
   * Used when unfreezing a plan (backward compat — will be removed in Step 5).
   */
  void demoteAllFrozen();

  // ── Thread binding ──────────────────────────────────────────────────────
  void bindToCurrentThread();
  void assertBoundThread() const;

  // ── Diagnostics ────────────────────────────────────────────────────────
  int countByOwnership(BufferOwnership type) const;
  int countByState(SlotLifecycleState state) const;

 private:
  int totalSlots_;
  SlotEntry* entries_;          // Owned heap array

  // Flat raw pointer array — kept in sync with entries_[i].array.
  // Eliminates the outputSlots_ / slotArrays_ duality.
  NDArray** rawArrays_;         // Owned heap array

  // Protected weight DataBuffers — never freed by plan cleanup.
  std::unordered_set<DataBuffer*> protectedWeightBuffers_;

  // Thread binding
  std::atomic<bool> bound_{false};
  std::thread::id ownerThread_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SLOT_ARRAY_H
