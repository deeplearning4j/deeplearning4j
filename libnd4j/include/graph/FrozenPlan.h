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

#ifndef LIBND4J_FROZEN_PLAN_H
#define LIBND4J_FROZEN_PLAN_H

#include <system/common.h>
#include <array/NDArray.h>
#include <graph/Context.h>

#include <cstdint>

namespace sd {
namespace graph {

// Forward declaration — full definition in NativeDynamicShapePlan.h
class NativeDynamicShapePlan;
enum class GraphExecutionMode : int;

/**
 * FrozenPlan — The single public entry point for DSP plan execution.
 *
 * This is the only type that Java/JNI callers interact with. It encapsulates
 * the entire plan lifecycle: warmup, compilation, capture, and replay.
 *
 * API contract:
 *   - First 1-2 execute() calls are the build phase (warmup + compile/capture)
 *   - Subsequent calls are sealed replay (fast path)
 *   - Callers never observe intermediate states — no setShapesFrozen(),
 *     no getPlanPhase(), no manual lifecycle management
 *
 * Diagnostics:
 *   - emitStateReport() fires DSP_DIAG_LIFECYCLE events for full plan/segment state
 *   - emitResourceReport() fires DSP_DIAG_MEMORY events for resource inventory
 *   - All state transitions are logged automatically when DSP_DIAG_LIFECYCLE is enabled
 *
 * Current implementation (Step 1): thin wrapper over NativeDynamicShapePlan.
 * Future steps will migrate internals into PlanTopology, SegmentExecutor,
 * ResourceBinder, and SlotArray.
 */
class SD_LIB_EXPORT FrozenPlan {
 public:
  /**
   * Create a FrozenPlan from a serialized binary plan.
   *
   * @param serializedBytes  Binary plan bytes (DSP1 format from Java DynamicShapePlan.serialize())
   * @param numBytes         Size of serialized data
   * @param graphExecutionMode  GraphExecutionMode ordinal (0=AUTO, 1=SLOT_BY_SLOT, etc.)
   * @return New FrozenPlan instance. Caller owns via the cache; do NOT delete directly.
   */
  static FrozenPlan* create(const void* serializedBytes, LongType numBytes,
                            int graphExecutionMode = 0);

  ~FrozenPlan();

  // Non-copyable, non-movable (cache holds the single instance)
  FrozenPlan(const FrozenPlan&) = delete;
  FrozenPlan& operator=(const FrozenPlan&) = delete;
  FrozenPlan(FrozenPlan&&) = delete;
  FrozenPlan& operator=(FrozenPlan&&) = delete;

  /**
   * Execute the plan.
   *
   * Internally handles warmup, compilation, capture, and replay phases.
   * The first few calls may be slower (build phase). Once sealed, every
   * call is a fast replay: sync inputs → arg table update → graph launch.
   *
   * @param context  Context with external inputs set
   * @param stream   CUDA stream (nullptr for CPU)
   * @return 0 on success, non-zero on failure
   */
  int execute(Context* context, void* stream);

  /**
   * Release GPU intermediates (graphs, workspaces) while keeping topology alive.
   * Next execute() will re-warm. Use between prefill/decode transitions.
   *
   * @return Number of intermediate NDArrays freed
   */
  int releaseGpuIntermediates();

  /**
   * Returns true once all segments have completed their build phase
   * and are in steady-state replay.
   */
  bool isSealed() const;

  /**
   * Emit structured state report via DSP_DIAG_LIFECYCLE.
   * Reports: buildPassCount, per-segment phase/backend/shapeKey/replayCount.
   * Only fires if DSP_DIAG_LIFECYCLE category is enabled.
   */
  void emitStateReport() const;

  /**
   * Emit resource inventory via DSP_DIAG_MEMORY.
   * Reports: streams, workspaces, arg tables, staging buffers, total bytes.
   * Only fires if DSP_DIAG_MEMORY category is enabled.
   */
  void emitResourceReport() const;

  // ─── Delegation to underlying plan (transitional API) ─────────────────
  // These methods expose the underlying NativeDynamicShapePlan for backward
  // compatibility during migration. They will be removed in Step 6.

  /** Get the underlying NativeDynamicShapePlan (for existing NativeOps functions). */
  NativeDynamicShapePlan* nativePlan() const { return plan_; }

  /** Structure identity hash for cache keying. */
  uint64_t identityFingerprint() const;

  /** Current plan phase code (for backward compatibility with getPlanPhase JNI). */
  int getPlanPhaseCode() const;

  /** Set shapes frozen (backward compat — will be removed in Step 6). */
  void setShapesFrozen(bool frozen);

  /** Clear shape caches (for session reset). */
  void clearShapeCaches();

  /** Force-clear all shape caches. */
  void clearAllShapeCachesForce();

  /** Get replay signature hash for a segment. */
  unsigned long long getReplaySignatureHash(int segIdx) const;

  /** Get replay unit count for a segment. */
  int getReplayUnitCount(int segIdx) const;

 private:
  FrozenPlan();  // Only created by create()

  // Step 1: delegates everything to the existing NativeDynamicShapePlan
  NativeDynamicShapePlan* plan_ = nullptr;

  // Build pass counter — tracks lifecycle phase:
  //   0 = needs warmup
  //   1 = needs compile/capture
  //   2+ = sealed (replay only)
  // In Step 1, this mirrors plan_->executeCount_ for diagnostics.
  // In Step 6, this replaces PlanPhase entirely.
  int buildPassCount_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FROZEN_PLAN_H
