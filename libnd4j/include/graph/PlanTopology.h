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

#ifndef LIBND4J_PLAN_TOPOLOGY_H
#define LIBND4J_PLAN_TOPOLOGY_H

#include <system/common.h>
#include <graph/NativeDynamicShapePlan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — full definitions in NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegmentDef;
struct LoopRegion;
enum class GraphExecutionMode : int;
enum class SelectedBackend : uint8_t;

/**
 * PlanTopology — Immutable shared structure for a compiled plan.
 *
 * Contains everything that NEVER changes after construction:
 *   - NativeSlot definitions (op hash, wiring, args, flags)
 *     NOTE: SlotState and generation are NOT here — those are mutable and
 *     live in SlotArray.
 *   - Segment definitions (boundaries, capturability, selected backend)
 *   - Release schedule (which slots can be freed after each step)
 *   - Requested output indices
 *   - External input metadata (names, isVariable flags)
 *   - Loop regions (for control flow)
 *   - Structure fingerprint (for plan cache keying)
 *
 * Shared via std::shared_ptr<const PlanTopology> — ref-counted, const-qualified.
 * Multiple FrozenPlan instances (e.g., multi-GPU serving) can share one topology.
 *
 * Thread safety: fully immutable after construction. Any number of threads can
 * read concurrently without synchronization.
 *
 * Current implementation (Step 4): wraps data extracted from NativeDynamicShapePlan.
 * Future: constructed directly from serialized bytes without intermediate plan.
 */
class SD_LIB_EXPORT PlanTopology {
 public:
  /**
   * Create a PlanTopology from an already-deserialized plan's data.
   * This is the Step 4 migration path: extract immutable data from
   * NativeDynamicShapePlan into PlanTopology.
   *
   * @param numSlots         Total number of op slots
   * @param numExternalInputs Number of external inputs (constants + variables + placeholders)
   * @param numRequestedOutputs Number of outputs requested by caller
   * @param requestedOutputIndices Slot indices of requested outputs
   * @param externalInputNames Names of external inputs (for diagnostics)
   * @param externalInputIsVariable Whether each external input can change between calls
   * @param numSegments      Number of graph segments
   * @param segmentDefs      Segment boundary definitions
   * @param releaseSchedule  Per-step release schedule (which slots to free)
   * @param loopRegions      While-loop region definitions
   * @param fingerprint      Structure identity hash
   * @param mode             GraphExecutionMode used to resolve backends
   */
  static std::shared_ptr<const PlanTopology> create(
      int numSlots,
      int numExternalInputs,
      int numRequestedOutputs,
      const int* requestedOutputIndices,
      const std::string* externalInputNames,
      const bool* externalInputIsVariable,
      int numSegments,
      const GraphSegmentDef* segmentDefs,
      const std::vector<std::vector<int>>& releaseSchedule,
      const std::vector<LoopRegion>& loopRegions,
      uint64_t fingerprint,
      GraphExecutionMode mode);

  ~PlanTopology();

  // ── Slot topology ──────────────────────────────────────────────────────
  int numSlots() const { return numSlots_; }
  int numExternalInputs() const { return numExternalInputs_; }
  int numRequestedOutputs() const { return numRequestedOutputs_; }

  /** Requested output slot indices. */
  const int* requestedOutputIndices() const { return requestedOutputIndices_; }
  int requestedOutputIndex(int i) const { return requestedOutputIndices_[i]; }

  // ── External input metadata ────────────────────────────────────────────
  const std::string& externalInputName(int i) const { return externalInputNames_[i]; }
  bool externalInputIsVariable(int i) const { return externalInputIsVariable_[i]; }

  // ── Segment topology ───────────────────────────────────────────────────
  int numSegments() const { return numSegments_; }
  const GraphSegmentDef& segmentDef(int i) const { return segmentDefs_[i]; }

  /** Get the segment index that contains a given slot. */
  int segmentForSlot(int slotIdx) const;

  // ── Release schedule ───────────────────────────────────────────────────

  /**
   * Slots to release after completing step `stepIdx`.
   * Returns empty vector if no slots to release at this step.
   */
  const std::vector<int>& releaseAtStep(int stepIdx) const;

  // ── Loop regions ───────────────────────────────────────────────────────
  int numLoopRegions() const { return static_cast<int>(loopRegions_.size()); }
  const LoopRegion& loopRegion(int i) const { return loopRegions_[i]; }
  bool hasControlFlow() const { return !loopRegions_.empty(); }

  // ── Identity ───────────────────────────────────────────────────────────

  /** Structure identity hash — used for plan cache keying. */
  uint64_t fingerprint() const { return fingerprint_; }

  /** The GraphExecutionMode that was used to resolve per-segment backends. */
  GraphExecutionMode executionMode() const { return mode_; }

  // ── Diagnostics ────────────────────────────────────────────────────────
  void emitReport() const;

 private:
  PlanTopology();  // Only created via create()

  int numSlots_;
  int numExternalInputs_;
  int numRequestedOutputs_;

  // Owned arrays
  int* requestedOutputIndices_;
  std::string* externalInputNames_;
  bool* externalInputIsVariable_;

  int numSegments_;
  GraphSegmentDef* segmentDefs_;  // Owned copy

  // Per-step release schedule: releaseSchedule_[stepIdx] = list of slot indices to free
  std::vector<std::vector<int>> releaseSchedule_;

  // Loop regions for control flow
  std::vector<LoopRegion> loopRegions_;

  // Structure fingerprint
  uint64_t fingerprint_;

  // Execution mode (for diagnostics and backend resolution reference)
  GraphExecutionMode mode_;

  // Empty vector returned by releaseAtStep() for out-of-range indices
  static const std::vector<int> EMPTY_RELEASE_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_PLAN_TOPOLOGY_H
