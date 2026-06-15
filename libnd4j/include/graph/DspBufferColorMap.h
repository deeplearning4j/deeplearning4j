/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#ifndef LIBND4J_DSP_BUFFER_COLOR_MAP_H
#define LIBND4J_DSP_BUFFER_COLOR_MAP_H

#include <system/common.h>
#include <array/NDArray.h>
#include <array/DataBuffer.h>
#include <graph/SlotBufferOwnership.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

// Forward declaration
class DspBufferPool;

// ═══════════════════════════════════════════════════════════════════════════════
// SlotLivenessData — immutable liveness analysis from NativePlanCompiler
// ═══════════════════════════════════════════════════════════════════════════════
//
// Persists the producer/lastConsumer step indices computed during plan
// compilation.  Lives on PlanDefinition (immutable, shared across per-thread
// plans).  Buffer coloring reads this to determine which slots have
// non-overlapping live ranges and can share physical buffers.

struct SlotLivenessData {
  int* producerStep = nullptr;       // Step index where each slot is produced
  int* lastConsumerStep = nullptr;   // Step index where each slot is last consumed
  int totalOutputSlots = 0;

  ~SlotLivenessData() {
    delete[] producerStep;
    delete[] lastConsumerStep;
  }

  // Non-copyable (owns arrays)
  SlotLivenessData() = default;
  SlotLivenessData(const SlotLivenessData&) = delete;
  SlotLivenessData& operator=(const SlotLivenessData&) = delete;

  void validate() const;
};

// ═══════════════════════════════════════════════════════════════════════════════
// DspBufferColorMap — compile-time buffer sharing via interval graph coloring
// ═══════════════════════════════════════════════════════════════════════════════
//
// Assigns a "color" to each eligible slot such that slots with the same color
// share a single physical DataBuffer.  Non-overlapping live ranges with
// compatible shapes get the same color.  This is analogous to register
// allocation — the "registers" are physical GPU buffers.
//
// Lifecycle integration:
//   - compute() runs after SHAPES_FROZEN, using liveness + ownership data
//   - apply() replaces per-slot buffers with shared color buffers (from pool)
//   - eject() undoes coloring gracefully (on shape change, OOM, validation fail)
//   - validate() checks invariants (every execute when debug mode is on)
//   - reset() clears assignments (on unfreeze/shape change)
//
// Safety:
//   - Views (VIEW_OF_SLOT) are NEVER colored — their parent's buffer is not ours
//   - View parents (slots with viewRefCount > 0) are NEVER colored
//   - Requested output slots are NEVER colored (user needs their dedicated buffer)
//   - Live ranges are EXTENDED to cover all VIEW_OF_SLOT children
//   - Any inconsistency throws std::runtime_error (caller can eject to recover)

class SD_LIB_EXPORT DspBufferColorMap {
 public:
  DspBufferColorMap();
  ~DspBufferColorMap();

  // Non-copyable, non-movable
  DspBufferColorMap(const DspBufferColorMap&) = delete;
  DspBufferColorMap& operator=(const DspBufferColorMap&) = delete;

  // ── Phases ─────────────────────────────────────────────────────────────

  /**
   * Assign colors from liveness + ownership data.
   * Groups slots by (shape, dtype, device), then applies greedy interval
   * coloring within each group.  Excludes views, view parents, requested
   * outputs, and non-SLOT_OWNED slots.
   *
   * Must be called after SHAPES_FROZEN when ownership is classified.
   */
  void compute(const SlotLivenessData& liveness,
               NDArray** outputSlots,
               int totalOutputSlots,
               const SlotBufferInfo* ownership,
               const std::unordered_set<int>& requestedOutputSlots);

  /**
   * Replace per-slot buffers with shared color buffers.
   * For each color, the "master" slot keeps its buffer.  All other slots
   * in the same color get a new NDArray wrapping the master's DataBuffer.
   * Acquires master buffers from the pool if needed (fresh allocation).
   *
   * @return Number of buffers consolidated (freed).
   */
  int apply(NDArray** outputSlots,
            SlotBufferInfo* ownership,
            std::unordered_set<NDArray*>& planOwnedArrays,
            DspBufferPool& pool);

  /**
   * Undo coloring — restore per-slot dedicated buffers.
   * For each colored slot sharing a buffer with another:
   *   1. Acquire a fresh dedicated buffer from pool
   *   2. Copy current data from shared buffer to new dedicated buffer
   *   3. Replace outputSlots[slotIdx] with new NDArray wrapping fresh buffer
   *   4. Update ownership and planOwnedArrays
   * Release old shared buffers back to pool.
   *
   * @return Number of slots restored to dedicated buffers.
   */
  int eject(NDArray** outputSlots,
            SlotBufferInfo* ownership,
            std::unordered_set<NDArray*>& planOwnedArrays,
            DspBufferPool& pool);

  // ── Validation ─────────────────────────────────────────────────────────

  /**
   * Validate all coloring invariants. Throws std::runtime_error on any
   * inconsistency.  Caller should catch and eject() to recover.
   *
   * Checks:
   *   - Same-color slots share the same DataBuffer address
   *   - No overlapping slots share a DataBuffer
   *   - VIEW_OF_SLOT parents are NOT colored
   *   - Uncolored SLOT_OWNED slots retain their original buffer
   *   - numColors matches distinct DataBuffer count among colored slots
   */
  void validate(NDArray** outputSlots,
                const SlotBufferInfo* ownership,
                int totalOutputSlots) const;

  // ── Introspection ──────────────────────────────────────────────────────

  bool isApplied() const { return applied_; }
  bool isComputed() const { return computed_; }
  int numColors() const { return numColors_; }
  int numColoredSlots() const { return numColoredSlots_; }
  int totalSlots() const { return totalOutputSlots_; }
  size_t estimatedBytesSaved() const { return bytesSaved_; }

  /**
   * Color assigned to slotIdx. Returns -1 if uncolored. Throws if OOB.
   */
  int colorOf(int slotIdx) const;

  /**
   * Number of slots sharing the given colorId.
   */
  int colorMemberCount(int colorId) const;

  /**
   * Buffer bytes for the given colorId.
   */
  size_t colorBufferBytes(int colorId) const;

  struct ColoringReport {
    int totalSlots;
    int coloredSlots;
    int uncoloredSlots;
    int numColors;
    size_t bytesBefore;
    size_t bytesAfter;
    size_t bytesSaved;
    int numShapeGroups;
    int largestGroupSize;
  };

  ColoringReport report() const;

  // ── Lifecycle ──────────────────────────────────────────────────────────

  /**
   * Clear color assignments. Called on unfreeze/shape change.
   * Does NOT restore buffers — call eject() first if applied_.
   */
  void reset();

 private:
  // Per-slot color assignment (-1 = uncolored)
  int* colorOf_ = nullptr;
  int totalOutputSlots_ = 0;

  // Aggregate stats
  int numColors_ = 0;
  int numColoredSlots_ = 0;
  size_t bytesBefore_ = 0;
  size_t bytesAfter_ = 0;
  size_t bytesSaved_ = 0;

  // State flags
  bool computed_ = false;
  bool applied_ = false;

  // Per-color metadata
  struct ColorInfo {
    int masterSlotIdx = -1;    // The slot that keeps its original buffer
    int memberCount = 0;       // Number of slots sharing this color
    size_t bufferBytes = 0;    // Size of the shared buffer
    DataBuffer* sharedBuffer = nullptr;  // Weak ref to the shared DataBuffer
  };
  std::vector<ColorInfo> colorInfos_;

  /**
   * Extend liveness ranges for view parents: if slot S has VIEW_OF_SLOT
   * children, S's effective lastConsumer must cover all children's ranges.
   */
  void computeViewExtendedLiveness(
      const SlotBufferInfo* ownership,
      int totalOutputSlots,
      const int* producerStep,
      const int* lastConsumerStep,
      int* effectiveLastConsumer) const;

  /**
   * Assert that two slots' live ranges do not overlap within a color.
   * Throws std::runtime_error if they overlap.
   */
  void assertNoOverlap(int slotA, int slotB,
                       const int* producerStep,
                       const int* effectiveLastConsumer,
                       int color) const;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_BUFFER_COLOR_MAP_H
