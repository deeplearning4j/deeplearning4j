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

#ifndef LIBND4J_PLAN_DEFINITION_H
#define LIBND4J_PLAN_DEFINITION_H

#include <system/common.h>
#include <graph/DspBufferColorMap.h>
#include <cstdint>
#include <atomic>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — full definitions in NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;
struct LoopRegion;

/**
 * PlanDefinition — Immutable plan structure shared across plan instances.
 *
 * Contains everything that does NOT change between executions:
 *   - Slot array (op hash, input/output wiring, args, flags)
 *   - Segment boundaries + capturability classification
 *   - Release schedule
 *   - Requested output indices
 *   - External input metadata (names, types, isVariable flags)
 *   - KV cache mapping definitions
 *   - Control flow structures (LoopRegions)
 *   - Backend priority order
 *
 * Multiple NativeDynamicShapePlan instances (e.g., one per thread) can share
 * a single PlanDefinition via ref-counting. The PlanDefinition is constructed
 * once during plan compilation and never modified afterward.
 *
 * Phase 3: Initially extracted as a thin wrapper. Fields will be moved from
 * NativeDynamicShapePlan incrementally to avoid a single massive change.
 */
class PlanDefinition {
 public:
  /**
   * Increment reference count. Called when a new plan instance starts
   * sharing this definition.
   */
  void addRef() { refCount_.fetch_add(1, std::memory_order_relaxed); }

  /**
   * Decrement reference count and delete if zero.
   * Returns true if this was the last reference (object deleted).
   */
  bool release() {
    int prev = refCount_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
      delete this;
      return true;
    }
    return false;
  }

  int refCount() const { return refCount_.load(std::memory_order_relaxed); }

  // ── Accessors (read-only after construction) ──────────────────────────

  int numSlots() const { return numSlots_; }
  int totalOutputSlots() const { return totalOutputSlots_; }
  int numExternalInputs() const { return numExternalInputs_; }
  int numRequestedOutputs() const { return numRequestedOutputs_; }
  const int* requestedOutputSlotIndices() const { return requestedOutputSlotIndices_; }

  const std::vector<std::string>& externalInputNames() const { return externalInputNames_; }
  const std::vector<bool>& externalInputIsVariable() const { return externalInputIsVariable_; }

  bool hasControlFlow() const { return hasControlFlow_; }
  int numLoopRegions() const { return numLoopRegions_; }

  const std::vector<std::string>& backendPriority() const { return backendPriority_; }

  /**
   * Slot liveness data (producer/lastConsumer step indices).
   * Populated by NativePlanCompiler. Used by DspBufferColorMap for
   * buffer coloring analysis. May be nullptr if liveness was not computed.
   */
  const SlotLivenessData* slotLiveness() const { return slotLiveness_; }

  // ── Builder (used by NativePlanCompiler / fromSerializedPlan) ──────────

  class Builder {
   public:
    Builder& setNumSlots(int n) { numSlots_ = n; return *this; }
    Builder& setTotalOutputSlots(int n) { totalOutputSlots_ = n; return *this; }
    Builder& setNumExternalInputs(int n) { numExternalInputs_ = n; return *this; }
    Builder& setNumRequestedOutputs(int n) { numRequestedOutputs_ = n; return *this; }
    Builder& setRequestedOutputSlotIndices(const int* indices, int count);
    Builder& setExternalInputNames(const std::vector<std::string>& names) {
      externalInputNames_ = names; return *this;
    }
    Builder& setExternalInputIsVariable(const std::vector<bool>& flags) {
      externalInputIsVariable_ = flags; return *this;
    }
    Builder& setHasControlFlow(bool v) { hasControlFlow_ = v; return *this; }
    Builder& setNumLoopRegions(int n) { numLoopRegions_ = n; return *this; }
    Builder& setBackendPriority(const std::vector<std::string>& p) {
      backendPriority_ = p; return *this;
    }
    Builder& setSlotLiveness(SlotLivenessData* liveness) {
      slotLiveness_ = liveness; return *this;
    }

    /**
     * Build the PlanDefinition. Caller receives a ref-counted pointer
     * (refCount starts at 1). Call release() when done.
     */
    PlanDefinition* build();

   private:
    int numSlots_ = 0;
    int totalOutputSlots_ = 0;
    int numExternalInputs_ = 0;
    int numRequestedOutputs_ = 0;
    int* requestedOutputSlotIndices_ = nullptr;
    std::vector<std::string> externalInputNames_;
    std::vector<bool> externalInputIsVariable_;
    bool hasControlFlow_ = false;
    int numLoopRegions_ = 0;
    std::vector<std::string> backendPriority_;
    SlotLivenessData* slotLiveness_ = nullptr;  // Ownership transferred to PlanDefinition
  };

 private:
  // Private constructor — use Builder
  PlanDefinition() : refCount_(1) {}
  ~PlanDefinition();

  // Non-copyable, non-movable
  PlanDefinition(const PlanDefinition&) = delete;
  PlanDefinition& operator=(const PlanDefinition&) = delete;

  std::atomic<int> refCount_;

  // ── Immutable plan data ───────────────────────────────────────────────

  int numSlots_ = 0;
  int totalOutputSlots_ = 0;
  int numExternalInputs_ = 0;
  int numRequestedOutputs_ = 0;
  int* requestedOutputSlotIndices_ = nullptr;  // Owned copy

  std::vector<std::string> externalInputNames_;
  std::vector<bool> externalInputIsVariable_;

  bool hasControlFlow_ = false;
  int numLoopRegions_ = 0;

  std::vector<std::string> backendPriority_;

  // Slot liveness data — immutable after construction, owned by PlanDefinition
  SlotLivenessData* slotLiveness_ = nullptr;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_PLAN_DEFINITION_H
