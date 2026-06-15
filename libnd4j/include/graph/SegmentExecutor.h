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

#ifndef LIBND4J_SEGMENT_EXECUTOR_H
#define LIBND4J_SEGMENT_EXECUTOR_H

#include <array/NDArray.h>
#include <graph/GraphReplayHandle.h>
#include <system/common.h>

#include <cstdint>
#include <memory>
#include <string>

namespace sd {
namespace graph {

// Forward declarations
class PlanTopology;
class ResourceBinder;
class SlotArray;
struct GraphSegmentDef;
enum class SelectedBackend : uint8_t;

/**
 * SegmentExecutor::Phase — 3-state machine (down from 7-state SegmentLifecycleState).
 *
 * State transitions (strict — no backward transitions, no temporary unfreeze):
 *
 *   BUILDING → SEALED   (capture/compile success — one-way, irreversible)
 *   BUILDING → FAILED   (permanent failure — terminal)
 *
 * BUILDING phase:
 *   - Warmup, compile, and capture all happen here
 *   - Slot states are mutable, shape inference runs
 *   - No frozen fast-path is used (eliminates temporary unfreeze entirely)
 *
 * SEALED phase:
 *   - Replay only. Replay handle, arg table, shape key are ALL immutable.
 *   - seal() is a one-way transition — no code path mutates state after this.
 *   - If shapes drift, this executor is DESTROYED and a new BUILDING executor
 *     takes its place (executor replacement pattern).
 *
 * FAILED phase:
 *   - Terminal. Slot-by-slot fallback.
 *   - Once failed, never retried (OOM handled differently — see below).
 *
 * Shape drift handling:
 *   Old design: demoteFrozenSlotStates → re-execute → restoreSlotStates (BUGGY)
 *   New design: destroy SEALED executor, create fresh BUILDING executor
 *               The old executor's replay handle is freed via its destructor.
 *               ResourceBinder reclaims the GPU resources.
 *
 * argTableStable elimination:
 *   Old design: boolean flag manually set/unset in 5+ locations (fragile)
 *   New design: address-key comparison at replay time:
 *     LongType currentKey = computeInputAddrKey(inputs, numInputs);
 *     if (currentKey != lastInputAddrKey_) { refresh; lastInputAddrKey_ = currentKey; }
 *   Zero-cost when stable (single pointer comparison). Self-healing when not.
 */
class SD_LIB_EXPORT SegmentExecutor {
 public:
  enum class Phase : uint8_t {
    BUILDING = 0,   // Warmup + compile + capture (mutable state)
    SEALED = 1,     // Replay only (fully immutable after seal())
    FAILED = 2,     // Terminal failure (slot-by-slot fallback)
  };

  /**
   * Create a SegmentExecutor for the given segment.
   *
   * @param topology    Shared plan topology (immutable)
   * @param segIdx      Index of this segment in the topology
   * @param resources   ResourceBinder for GPU resource management
   */
  SegmentExecutor(std::shared_ptr<const PlanTopology> topology,
                  int segIdx,
                  ResourceBinder* resources);
  ~SegmentExecutor();

  // Non-copyable, non-movable (owns replay handle)
  SegmentExecutor(const SegmentExecutor&) = delete;
  SegmentExecutor& operator=(const SegmentExecutor&) = delete;
  SegmentExecutor(SegmentExecutor&&) = delete;
  SegmentExecutor& operator=(SegmentExecutor&&) = delete;

  // ── State queries ──────────────────────────────────────────────────────
  Phase phase() const { return phase_; }
  bool isBuilding() const { return phase_ == Phase::BUILDING; }
  bool isSealed() const { return phase_ == Phase::SEALED; }
  bool isFailed() const { return phase_ == Phase::FAILED; }
  int segmentIndex() const { return segIdx_; }
  int executionCount() const { return executionCount_; }

  /** Sealed shape key (only valid after seal()). */
  LongType sealedShapeKey() const { return sealedShapeKey_; }

  /** Last input address key (for drift detection). */
  LongType lastInputAddrKey() const { return lastInputAddrKey_; }

  /** Backend that compiled/captured this segment. */
  const std::string& backendName() const { return backendName_; }

  /** Selected backend from topology. */
  SelectedBackend selectedBackend() const;

  // ── Build phase operations ─────────────────────────────────────────────

  /**
   * Execute warmup pass: slot-by-slot execution to populate shape caches.
   * Only valid during BUILDING phase.
   *
   * @param inputs     External input arrays
   * @param numInputs  Number of external inputs
   * @param slots      SlotArray for output storage
   * @param stream     CUDA stream (nullptr for CPU)
   * @return 0 on success, non-zero on failure
   */
  int buildWarmup(NDArray** inputs, int numInputs, SlotArray* slots, void* stream);

  /**
   * Execute compile pass: call backend->compileSegment().
   * Only valid during BUILDING phase after warmup.
   *
   * @param inputs     External input arrays
   * @param numInputs  Number of external inputs
   * @param slots      SlotArray with warmup-populated outputs
   * @param stream     CUDA stream (nullptr for CPU)
   * @return 0 on success, non-zero on failure (transitions to FAILED)
   */
  int buildCompile(NDArray** inputs, int numInputs, SlotArray* slots, void* stream);

  /**
   * Execute capture pass: record operations into graph for replay.
   * On success, calls seal() to transition to SEALED.
   * Only valid during BUILDING phase after compile.
   *
   * @param inputs     External input arrays
   * @param numInputs  Number of external inputs
   * @param slots      SlotArray with compiled outputs
   * @param stream     CUDA stream (nullptr for CPU)
   * @return 0 on success, non-zero on failure (transitions to FAILED)
   */
  int buildCapture(NDArray** inputs, int numInputs, SlotArray* slots, void* stream);

  // ── Sealed phase operations ────────────────────────────────────────────

  /**
   * Replay the captured graph.
   * Only valid during SEALED phase.
   *
   * Internally:
   *   1. Compute input address key
   *   2. If key differs from lastInputAddrKey_ → refresh arg table
   *   3. Launch graph replay
   *
   * @param inputs     External input arrays
   * @param numInputs  Number of external inputs
   * @param slots      SlotArray (for arg table refresh if needed)
   * @param stream     CUDA stream (nullptr for CPU)
   * @return 0 on success, non-zero on failure
   */
  int replay(NDArray** inputs, int numInputs, SlotArray* slots, void* stream);

  // ── Lifecycle transitions ──────────────────────────────────────────────

  /**
   * Seal the executor — one-way transition from BUILDING to SEALED.
   * After this call, no mutable state is ever touched again.
   * The replay handle and shape key become immutable.
   *
   * @param shapeKey  The shape key computed during build
   */
  void seal(LongType shapeKey);

  /**
   * Mark as permanently failed — one-way transition to FAILED.
   * After this, only slot-by-slot execution is used for this segment.
   *
   * @param reason  Human-readable failure reason (for diagnostics)
   */
  void markFailed(const std::string& reason);

  // ── Shape drift detection ──────────────────────────────────────────────

  /**
   * Check if the current shape key matches the sealed shape key.
   * If shapes have drifted, the caller should destroy this executor
   * and create a new BUILDING one (executor replacement pattern).
   *
   * @param currentShapeKey  Shape key computed from current inputs
   * @return true if shapes match (safe to replay), false if drifted
   */
  bool shapesMatch(LongType currentShapeKey) const;

  // ── Address key computation ────────────────────────────────────────────

  /**
   * Compute the input address key from external input pointers.
   * Used for zero-cost arg table stability check (replaces argTableStable boolean).
   *
   * @param inputs     External input arrays
   * @param numInputs  Number of external inputs
   * @return FNV-1a hash of input buffer addresses
   */
  static LongType computeInputAddrKey(NDArray** inputs, int numInputs);

  // ── Replay handle access (for backward compat during migration) ────────
  GraphReplayHandle* replayHandle() const { return replayHandle_.get(); }
  void setReplayHandle(std::unique_ptr<GraphReplayHandle> handle);

  // ── Diagnostics ────────────────────────────────────────────────────────

  /** JNI-compatible phase code (matches old ExecutionPhase ordinals). */
  int getPhaseCode() const;

  /** Display name for current phase. */
  const char* phaseName() const;

  /** Emit segment state via DSP_DIAG_LIFECYCLE. */
  void emitReport() const;

 private:
  // Immutable topology reference
  std::shared_ptr<const PlanTopology> topology_;
  int segIdx_;

  // Resource binder (not owned — lifetime managed by FrozenPlan)
  ResourceBinder* resources_;

  // ── Mutable state (only during BUILDING) ───────────────────────────────
  Phase phase_ = Phase::BUILDING;
  int executionCount_ = 0;

  // ── Sealed state (immutable after seal()) ──────────────────────────────
  LongType sealedShapeKey_ = 0;
  LongType lastInputAddrKey_ = 0;
  std::string backendName_;

  // Platform-agnostic replay handle:
  //   CUDA: CudaGraphReplayHandle (wraps cudaGraph_t/cudaGraphExec_t)
  //   CPU: FunctionalReplayHandle (cached op dispatch)
  std::unique_ptr<GraphReplayHandle> replayHandle_;

  // Failure reason (set on transition to FAILED)
  std::string failureReason_;

  // Consecutive stable-replay counter (for diagnostics)
  int stableReplayCount_ = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SEGMENT_EXECUTOR_H
