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

/**
 * PlanExecutionContext — per-execute() invocation state.
 *
 * Replaces the ad-hoc void* executionStatePtr / inline ExecutionState struct
 * that previously carried only a DspStreamGuard through execute()'s lifecycle.
 *
 * This struct consolidates ALL state that flows across execution phases:
 *   - CUDA stream handles and cross-stream sync infrastructure
 *   - Execution identity (frozen state, exec count at entry)
 *   - Precomputed derived booleans (computed once, read everywhere)
 *   - Phase dispatch coordination flags
 *   - Timing instrumentation points
 *   - Sync tracking (which syncs have happened, is GPU data safe to read)
 *   - Diagnostic tracing (output fingerprints, segment trace)
 *   - DSP diagnostic lifecycle (beginStep/endStep tracking)
 *
 * Ownership: heap-allocated by platformBeginExecution(), passed through
 * execute() as a typed pointer (cast from void* at the header boundary),
 * freed by platformEndExecution(). Lifetime = one execute() call.
 *
 * This header is intentionally NOT included by NativeDynamicShapePlan.h.
 * Only .cpp/.cu implementation files include it. The main header keeps
 * void* signatures for platformBeginExecution/platformEndExecution,
 * and .cpp/.cu files cast to PlanExecutionContext* internally.
 * This avoids header rebuild cascades.
 */

#ifndef LIBND4J_PLAN_EXECUTION_CONTEXT_H
#define LIBND4J_PLAN_EXECUTION_CONTEXT_H

#include <system/common.h>
#include <graph/ModeContract.h>
#include <graph/DspDiagnostics.h>

#include <chrono>
#include <memory>
#include <cstdint>
#include <unordered_map>

namespace sd {
namespace graph {

/**
 * SyncLevel - what level of GPU ordering has been performed.
 *
 * Used to determine whether GPU memory is safe to read from the host.
 * Higher values subsume lower: FULL_DEVICE implies STREAM implies EVENT.
 */
enum class SyncLevel : uint8_t {
  NONE = 0,          // No sync — GPU data may be in-flight
  EVENT = 1,         // Event-based ordering — other CUDA streams will wait,
                     //   but HOST reads are NOT safe
  STREAM = 2,        // Legacy blocking stream-drain state.
                     // Host reads on DSP output are safe only for legacy callers.
  FULL_DEVICE = 3,   // Legacy blocking device-drain state.
};

/**
 * PreReplaySyncPhase — ordered state machine for per-step sync deduplication.
 *
 * Replaces the scattered booleans (crossStreamSynced, extInputsSynced,
 * stagingBuffersSynced) with a single enum that enforces ordering:
 *
 *   UNSYNCED ──► CROSS_STREAM_DONE ──► EXT_INPUTS_DONE ──► STAGING_DONE
 *
 * Each phase subsumes all prior phases. STAGING_DONE implies cross-stream
 * ordering and H2D sync are both complete. The ordering is NOT arbitrary —
 * D2D staging reads from device buffers that were just H2D-synced, and H2D
 * relies on cross-stream ordering to see Java-side .assign() writes.
 *
 * Invalid transitions (e.g., jumping from UNSYNCED to STAGING_DONE) are
 * caught by assertions. This is the single source of truth for "what sync
 * work has been done this execute() step."
 *
 * Lifetime: reset to UNSYNCED at the start of each execute() call (when
 * PlanExecutionContext is created fresh or reset for reuse).
 */
enum class PreReplaySyncPhase : uint8_t {
  UNSYNCED          = 0,  // Nothing synced yet this step
  CROSS_STREAM_DONE = 1,  // Event: default stream → DSP stream ordering complete
  EXT_INPUTS_DONE   = 2,  // H2D sync of variable external inputs complete
  STAGING_DONE      = 3,  // D2D copy into plan-owned staging buffers complete
};

/**
 * ExecTarget — what execution mode this step's sync/staging should prepare for.
 *
 * Set ONCE per segment dispatch, read by performPreReplaySync to decide
 * which sync steps to run and what to return:
 *
 *   SBS_ON_LC_STREAM — Slot-by-slot on LaunchContext stream.
 *     Needs: H2D for variable inputs (isPrimaryActual guard).
 *     No staging (ops read raw ext arrays on LC stream).
 *     No cross-stream ordering needed (assign + ops = same stream).
 *     Returns: raw external arrays.
 *
 *   GRAPH_CAPTURE — Pre-capture setup on DSP stream.
 *     Needs: H2D, cross-stream LC->DSP, D2D staging, same-stream ordering.
 *     Returns: staged arrays (graph bakes in staging addresses).
 *
 *   GRAPH_REPLAY — Graph replay on DSP stream.
 *     Needs: H2D, cross-stream LC→DSP, D2D staging.
 *     Async replay reads from staging after event/same-stream ordering.
 *     Returns: staged arrays.
 */
enum class ExecTarget : uint8_t {
  SBS_ON_LC_STREAM = 0,  // Slot-by-slot: H2D only, no staging, raw ext arrays
  GRAPH_CAPTURE    = 1,  // Pre-capture: H2D + cross-stream + staging + stream sync
  GRAPH_REPLAY     = 2,  // Graph replay: H2D + cross-stream + staging (async)
};

struct PlanExecutionContext {
  // ══════════════════════════════════════════════════════════════════════
  // Stream/event state (void* on all platforms; cast to cudaStream_t /
  // cudaEvent_t in CUDA implementation files).
  // Set by platformBeginExecution. Available to all subsequent phases.
  // ══════════════════════════════════════════════════════════════════════
  void* dspStream = nullptr;        // The DSP execution stream (from caller)
  void* lcDefaultStream = nullptr;  // LaunchContext default exec stream
  void* streamGuard = nullptr;      // Owned, explicit delete in platformEndExecution
  int deviceId = 0;                 // CUDA device captured at platformBeginExecution

  /**
   * Cross-stream sync event for this execution.
   * Created by platformBeginExecution(), destroyed by platformEndExecution().
   * Used by performPreReplaySync() to record on the default stream and wait
   * on the DSP stream, establishing ordering between Java-side .assign()
   * ops and DSP graph replay. Lifetime tied to execution context, not thread.
   * void* on all platforms; cast to cudaEvent_t in CUDA implementation files.
   */
  void* crossStreamEvent = nullptr;

  /**
   * FNV-1a fingerprints of variable external inputs from the previous
   * compositeReplay call. Detects stuck inputs that never change between
   * execution steps. Replaces tl_prevVariableFingerprints (was thread_local
   * in _gpubackend.cpp) so state is per-plan-execution, not per-thread.
   * Keyed by external-input index.
   */
  std::unordered_map<int, uint64_t> prevVariableFingerprints;

  // ══════════════════════════════════════════════════════════════════════
  // Execution identity (snapshot at entry)
  // Captured at the start of execute() before any mutations.
  // Use these when you need "what was true when this execute() began"
  // rather than the live plan fields which may advance mid-execution.
  // ══════════════════════════════════════════════════════════════════════
  int execCount = 0;          // executeCount_ at start of this execute()
  bool frozen = false;        // planLifecycle_.isShapesFrozen() at start of this execute()

  // ══════════════════════════════════════════════════════════════════════
  // Precomputed derived state (computed once at execute() entry)
  //
  // These eliminate the scattered re-derivation of the same conditions
  // across execute(), platform methods, and segment dispatch.
  // All are set by populateDerivedState() called from execute().
  // ══════════════════════════════════════════════════════════════════════

  /**
   * Frozen steady-state: frozen && execCount > 1.
   * Controls: event-based sync (vs full sync), stale-buffer scan skip,
   *           shape key reuse, weight skip in performPreReplaySync H2D.
   */
  bool isFrozenSteadyState = false;

  /**
   * First frozen warmup: frozen && execCount == 0.
   * Controls: phaseWarmup dispatch, skip phaseCompile, segment state reset.
   */
  bool isFirstFrozenWarmup = false;

  /**
   * Warmup or capture: !frozen || execCount <= 1.
   * Controls: warmup/capture event ordering in begin/end.
   * Inverse of isFrozenSteadyState when frozen=true, but also true when !frozen.
   */
  bool needsFullSync = true;

  /**
   * Master gate for Triton CUDA graph capture/replay.
   * Derived from: tritonGraphCapture() && frozen && !tritonSkipKernels().
   * Used in executeSegmentWithGpuGraph and segDispatchCaptureOrDirect.
   */
  bool allowGraphCaptureReplay = false;

  /**
   * Whether external input variable/weight classification is available.
   * Derived from: frozen && !externalInputIsVariable_.empty().
   * Controls performPreReplaySync H2D behavior and frozen fast path.
   */
  bool useVariableFilter = false;

  /**
   * Whether tritonSkipKernels forced slot-by-slot override.
   * When true, graphExecutionMode_ was overridden to GEM_SLOT_BY_SLOT.
   */
  bool forcedSlotBySlot = false;

  /**
   * Normal decode path: not first-frozen-warmup and not forced-slot-by-slot.
   * Complement of isFirstFrozenWarmup and forcedSlotBySlot.
   * Controls: per-segment graph replay or capture.
   */
  bool isReplay = false;

  /**
   * Whether VERIFY diagnostics should fire for this execution.
   * Derived from: DSP_DIAG_ENABLED(VERIFY) && withinExecLimit(execCount).
   * Avoids re-checking at every diagnostic site.
   */
  bool diagVerifyEnabled = false;

  /**
   * Whether any diagnostic category is enabled for this execution.
   * Quick check to avoid string formatting overhead in hot paths.
   */
  bool diagAnyEnabled = false;

  /**
   * Whether triton kernel verification is enabled this step.
   * Derived from: Environment::getInstance().tritonVerifyKernels().
   * Read in compositeReplay, segment dispatch, arg table refresh.
   */
  bool tritonVerifyEnabled = false;

  /**
   * The graph execution mode for this execution.
   * Snapshot of graphExecutionMode_ at entry (after tritonSkipKernels override).
   * Avoids reading the mutable plan field mid-execution.
   */
  int graphExecutionMode = 0;  // GraphExecutionMode value

  // ══════════════════════════════════════════════════════════════════════
  // Sync tracking
  //
  // Records the highest sync level achieved during this execute() call.
  // Used by diagnostic functions to determine whether GPU data is safe
  // to read from the host (avoiding the race condition where
  // platformDumpLogitsArgmax reads zeros from in-flight GPU writes).
  // ══════════════════════════════════════════════════════════════════════

  /** Current sync level — set by platform begin/end and segment dispatch. */
  SyncLevel currentSyncLevel = SyncLevel::NONE;

  /** Legacy blocking stream-drain counter during this execute(). */
  int streamSyncCount = 0;

  /** Number of event-based sync orderings recorded. */
  int eventSyncCount = 0;

  // ══════════════════════════════════════════════════════════════════════
  // Per-step sync state machine
  //
  // Plan-level sync operations (cross-stream ordering, H2D ext input sync,
  // D2D staging copy) only need to run once per step, not per segment.
  // The sync phase tracks what has been done this step. Each transition
  // subsumes all prior phases. Enforced ordering prevents bugs where D2D
  // staging reads stale data because cross-stream ordering was skipped.
  //
  // No manual reset needed for fresh contexts (default = UNSYNCED).
  // For reused contexts, resetSyncPhase() resets to UNSYNCED.
  // ══════════════════════════════════════════════════════════════════════

  /** Current sync phase — single source of truth for per-step sync dedup. */
  PreReplaySyncPhase syncPhase = PreReplaySyncPhase::UNSYNCED;

  /**
   * Execution target for the current segment dispatch.
   * Set by dispatchSegment / platformTryFrozenFastPath / executeSegmentWithGraph
   * BEFORE calling performPreReplaySync. Read by performPreReplaySync to decide
   * which sync steps to run (H2D only vs H2D+staging) and what to return.
   *
   * Default: SBS_ON_LC_STREAM (safest — H2D only, no staging).
   */
  ExecTarget execTarget = ExecTarget::SBS_ON_LC_STREAM;

  // ── Sync phase queries ────────────────────────────────────────────────
  /** Cross-stream event ordering has been recorded this step. */
  SD_INLINE bool isCrossStreamSynced() const {
    return syncPhase >= PreReplaySyncPhase::CROSS_STREAM_DONE;
  }

  /** Variable external inputs have been H2D-synced this step. */
  SD_INLINE bool isExtInputsSynced() const {
    return syncPhase >= PreReplaySyncPhase::EXT_INPUTS_DONE;
  }

  /** D2D staging copy into plan-owned buffers is complete this step. */
  SD_INLINE bool isStagingBuffersSynced() const {
    return syncPhase >= PreReplaySyncPhase::STAGING_DONE;
  }

  // ── Sync phase transitions (validated) ────────────────────────────────
  /** Record cross-stream sync done. Must be called from UNSYNCED. */
  SD_INLINE void markCrossStreamSynced() {
    assert(syncPhase == PreReplaySyncPhase::UNSYNCED &&
           "markCrossStreamSynced: expected UNSYNCED");
    syncPhase = PreReplaySyncPhase::CROSS_STREAM_DONE;
  }

  /** Record ext inputs H2D sync done. Must be at CROSS_STREAM_DONE. */
  SD_INLINE void markExtInputsSynced() {
    assert(syncPhase == PreReplaySyncPhase::CROSS_STREAM_DONE &&
           "markExtInputsSynced: expected CROSS_STREAM_DONE");
    syncPhase = PreReplaySyncPhase::EXT_INPUTS_DONE;
  }

  /** Record staging D2D copy done. Must be at EXT_INPUTS_DONE. */
  SD_INLINE void markStagingBuffersSynced() {
    assert(syncPhase == PreReplaySyncPhase::EXT_INPUTS_DONE &&
           "markStagingBuffersSynced: expected EXT_INPUTS_DONE");
    syncPhase = PreReplaySyncPhase::STAGING_DONE;
  }

  /** Reset sync phase to UNSYNCED (for reused contexts at step start). */
  SD_INLINE void resetSyncPhase() {
    syncPhase = PreReplaySyncPhase::UNSYNCED;
  }

  /** Display name for the current exec target. */
  SD_INLINE const char* execTargetName() const {
    switch (execTarget) {
      case ExecTarget::SBS_ON_LC_STREAM: return "SBS_ON_LC_STREAM";
      case ExecTarget::GRAPH_CAPTURE:    return "GRAPH_CAPTURE";
      case ExecTarget::GRAPH_REPLAY:     return "GRAPH_REPLAY";
      default:                           return "UNKNOWN";
    }
  }

  /** Display name for the current sync phase. */
  SD_INLINE const char* syncPhaseName() const {
    switch (syncPhase) {
      case PreReplaySyncPhase::UNSYNCED:          return "UNSYNCED";
      case PreReplaySyncPhase::CROSS_STREAM_DONE: return "CROSS_STREAM_DONE";
      case PreReplaySyncPhase::EXT_INPUTS_DONE:   return "EXT_INPUTS_DONE";
      case PreReplaySyncPhase::STAGING_DONE:      return "STAGING_DONE";
      default:                                    return "UNKNOWN";
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Execution flow tracking
  //
  // Records WHAT happened during this execute() call and in what order.
  // Every significant state mutation (auto-seal, frozen constant detection,
  // phase transition, compilation) is logged here with a monotonic sequence
  // number so post-mortem analysis shows the exact flow.
  //
  // FlowEvent is a compact record — no heap strings. Each event type has
  // a fixed meaning; numeric fields carry context-dependent detail.
  // ══════════════════════════════════════════════════════════════════════

  enum class FlowEventType : uint8_t {
    EXECUTE_ENTRY = 0,          // execute() entered; detail1=execCount, detail2=frozen
    AUTO_SEAL_FIRED = 1,        // auto-seal set planLifecycle_.isShapesFrozen()=true; detail1=old_execCount, detail2=new_execCount
    RESEGMENT = 2,              // resegmentForFreeze(); detail1=old_seg_count, detail2=new_seg_count
    FROZEN_CONST_DETECT = 3,    // detectFrozenConstants(); detail1=frozen_count, detail2=total_slots
    PHASE_COMPILE = 4,          // phaseCompile(); detail1=num_segments_compiled
    PHASE_DISPATCH = 5,         // phase dispatch; detail1=graphExecutionMode (0=REPLAY,1=SLOT_BY_SLOT), isFirstFrozenWarmup if warmup
    SLOT_EXEC_FAIL = 6,         // slot execution failed; detail1=slotIdx, detail2=status
    SLOT_FROZEN_SKIP = 7,       // frozen constant slot skipped; detail1=slotIdx
    PHASE_TRANSITION = 8,       // planPhase_ changed; detail1=old_phase, detail2=new_phase
    EXEC_COUNT_INC = 9,         // executeCount_ incremented; detail1=old, detail2=new
    DERIVED_STATE_REFRESH = 10, // populateDerivedState re-called after mutation
  };

  struct FlowEvent {
    FlowEventType type;
    int detail1 = 0;
    int detail2 = 0;
  };

  static constexpr int kMaxFlowEvents = 64;
  FlowEvent flowEvents[kMaxFlowEvents];
  int flowEventCount = 0;

  /** Record a flow event. Silently drops if buffer full. */
  SD_INLINE void recordFlow(FlowEventType type, int d1 = 0, int d2 = 0) {
    if (flowEventCount < kMaxFlowEvents) {
      flowEvents[flowEventCount++] = {type, d1, d2};
    }
  }

  /** Get human-readable name for a flow event type. */
  static SD_INLINE const char* flowEventName(FlowEventType type) {
    switch (type) {
      case FlowEventType::EXECUTE_ENTRY:        return "EXECUTE_ENTRY";
      case FlowEventType::AUTO_SEAL_FIRED:      return "AUTO_SEAL_FIRED";
      case FlowEventType::RESEGMENT:            return "RESEGMENT";
      case FlowEventType::FROZEN_CONST_DETECT:  return "FROZEN_CONST_DETECT";
      case FlowEventType::PHASE_COMPILE:        return "PHASE_COMPILE";
      case FlowEventType::PHASE_DISPATCH:       return "PHASE_DISPATCH";
      case FlowEventType::SLOT_EXEC_FAIL:       return "SLOT_EXEC_FAIL";
      case FlowEventType::SLOT_FROZEN_SKIP:     return "SLOT_FROZEN_SKIP";
      case FlowEventType::PHASE_TRANSITION:     return "PHASE_TRANSITION";
      case FlowEventType::EXEC_COUNT_INC:       return "EXEC_COUNT_INC";
      case FlowEventType::DERIVED_STATE_REFRESH: return "DERIVED_STATE_REFRESH";
      default:                                  return "UNKNOWN";
    }
  }

  /** Log all recorded flow events via DSP_DIAG. Called on error or at execution end. */
  SD_INLINE void dumpFlowLog(int execCount) {
    DSP_DIAG(EXECUTE, "FLOW_LOG exec=%d (%d events):", execCount, flowEventCount);
    for (int i = 0; i < flowEventCount; i++) {
      auto& e = flowEvents[i];
      DSP_DIAG(EXECUTE, "  FLOW[%d] %s d1=%d d2=%d",
               i, flowEventName(e.type), e.detail1, e.detail2);
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // Frozen constant tracking
  //
  // Populated by detectFrozenConstants() so downstream code and
  // diagnostics can inspect what was frozen and why.
  // ══════════════════════════════════════════════════════════════════════

  int frozenConstCount = 0;      // Slots marked FROZEN_CONSTANT
  int shapeOnlyTraitCount = 0;   // Slots with OP_TRAIT_SHAPE_ONLY_OUTPUT
  int valueIndepCount = 0;       // Value-independent slots (subset of frozen)
  int viewAliasUnfrozen = 0;     // Frozen slots un-frozen due to buffer aliasing
  int javaManagedSlots = 0;      // Output slots not produced by any plan op

  // ══════════════════════════════════════════════════════════════════════
  // Segment execution tracking
  //
  // Per-execution counters for tracing what happened at the segment level.
  // Set during segment dispatch, read by post-execution diagnostics.
  // ══════════════════════════════════════════════════════════════════════

  int segmentsTotal = 0;          // Total segments dispatched
  int segmentsWarmup = 0;         // Segments that ran warmup (slot-by-slot for shapes)
  int segmentsCaptured = 0;       // Segments that completed CUDA graph capture
  int segmentsReplayed = 0;       // Segments that ran graph replay
  int segmentsSlotBySlot = 0;     // Segments that ran slot-by-slot (non-capturable)
  int segmentsDirect = 0;         // Segments that ran direct Triton execution
  int segmentsFailed = 0;         // Segments that hit permanent failure
  int segmentsInvalidated = 0;    // Segments invalidated for rebuild this step

  // ══════════════════════════════════════════════════════════════════════
  // Output tracing
  //
  // Records output state for debugging accuracy issues.
  // Only populated when diagVerifyEnabled is true.
  // ══════════════════════════════════════════════════════════════════════

  /** Argmax index of the logits output (largest FLOAT32 requested output). -1 if not computed. */
  int logitsArgmaxIdx = -1;

  /** Max value in the logits output. 0 if not computed. */
  float logitsArgmaxVal = 0.0f;

  /** Whether logits output was all zeros (indicates a bug or race condition). */
  bool logitsAllZeros = false;

  /** Whether the logits diagnostic successfully synced the stream before reading. */
  bool logitsSyncedBeforeRead = false;

  // ══════════════════════════════════════════════════════════════════════
  // Timing instrumentation
  //
  // All time_points default to epoch (zero). Only populated when
  // timingEnabled is true. Use now() helper to avoid scattered ternaries.
  // ══════════════════════════════════════════════════════════════════════
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point t0;            // Execution start
  Clock::time_point tSegsDone;     // After all segment execution
  Clock::time_point tOutputsDone;  // After output collection
  Clock::time_point tFlushDone;    // After flush
  bool timingEnabled = false;

  // ══════════════════════════════════════════════════════════════════════
  // DSP diagnostics lifecycle
  //
  // Tracks whether DspDiagnostics::beginStep() was called for this
  // execution, ensuring endStep() is always called on all exit paths
  // (including early returns). Without this, a missed endStep() leaves
  // the diagnostic ring buffer in an inconsistent state.
  // ══════════════════════════════════════════════════════════════════════
  bool diagStepStarted = false;

  // ══════════════════════════════════════════════════════════════════════
  // Convenience methods
  // ══════════════════════════════════════════════════════════════════════

  /**
   * Returns current time if timing is enabled, otherwise a zero time_point.
   * Avoids the scattered `timingEnabled ? Clock::now() : Clock::time_point{}`
   * pattern throughout execute().
   */
  SD_INLINE Clock::time_point now() const {
    return timingEnabled ? Clock::now() : Clock::time_point{};
  }

  /**
   * Begin DSP diagnostic step tracking. Safe to call once per execute().
   * Sets diagStepStarted = true so cleanup paths know to call endDiag().
   */
  SD_INLINE void beginDiag(int execCount) {
    DspDiagnostics::getInstance().beginStep(execCount);
    diagStepStarted = true;
  }

  /**
   * End DSP diagnostic step tracking. No-op if beginDiag() was not called.
   * Safe to call multiple times (only the first call has effect).
   */
  SD_INLINE void endDiag(int execCount) {
    if (diagStepStarted) {
      DspDiagnostics::getInstance().endStep(execCount);
      diagStepStarted = false;
    }
  }

  /**
   * Record a stream sync event. Updates currentSyncLevel and counter.
   */
  SD_INLINE void recordStreamSync() {
    currentSyncLevel = SyncLevel::STREAM;
    streamSyncCount++;
  }

  /**
   * Record an event-based sync ordering. Updates level only if no stronger sync exists.
   */
  SD_INLINE void recordEventSync() {
    if (currentSyncLevel < SyncLevel::EVENT) {
      currentSyncLevel = SyncLevel::EVENT;
    }
    eventSyncCount++;
  }

  /**
   * Record a full device sync. Updates to strongest level.
   */
  SD_INLINE void recordDeviceSync() {
    currentSyncLevel = SyncLevel::FULL_DEVICE;
    streamSyncCount++;
  }

  /**
   * Check whether host reads of GPU data are safe (stream or device sync done).
   */
  SD_INLINE bool isHostReadSafe() const {
    return currentSyncLevel >= SyncLevel::STREAM;
  }

  /**
   * Populate all derived state fields from plan member values.
   * Called once at execute() entry after execCount and frozen are set.
   *
   * Parameters come from the plan's member fields — this method is called
   * from execute() which has direct access to them.
   */
  SD_INLINE void populateDerivedState(
      bool shapesFrozen, int executeCount, int gemMode,
      bool tritonSkipKernels, bool tritonGraphCapture,
      bool tritonVerify, bool hasVariableList,
      bool execTimingEnabled,
      bool anySegmentInWarmup) {

    // Snapshot execution identity
    execCount = executeCount;
    frozen = shapesFrozen;

    // Derived booleans — computed once, read everywhere.
    // isFirstFrozenWarmup fires phaseWarmup on the FIRST frozen execution ONLY, and
    // that is deliberate, not a bug: phaseWarmup RESETS every segment's executionCount
    // to 0 (it rebuilds slot/shape state), so it must run exactly once. CUDA-graph
    // capture does NOT complete during warmup — it accumulates across the subsequent
    // phaseReplay executions: each executeSegmentSlotBySlot bumps segment executionCount,
    // and a segment captures+seals when it reaches captureMinExec (=2). Lifecycle:
    //   exec0 -> WARMUP (reset+compile)   exec1 -> REPLAY (executionCount->2, capture)
    //   exec2+ -> REPLAY (graph replay).
    // A "persistent" warmup keyed on anySegmentNeedsWarmup() would re-reset executionCount
    // every step, so capture could NEVER reach captureMinExec — do not do that.
    // anySegmentInWarmup (== anySegmentNeedsWarmup(), the capture window) is the SINGLE
    // source of truth for the SYNC booleans below; it must NOT gate the warmup trigger.
    isFirstFrozenWarmup = shapesFrozen && executeCount == 0;
    isFrozenSteadyState = shapesFrozen && executeCount > 1 && !anySegmentInWarmup;
    needsFullSync = !shapesFrozen || executeCount <= 1 || anySegmentInWarmup;

    // Mode contract — computed once, read by the slot-by-slot override and capture gate.
    const ModeContract modeContract = ModeContract::forMode(gemMode);

    // tritonSkipKernels override
    forcedSlotBySlot = tritonSkipKernels && !modeContract.isSlotBySlot;
    graphExecutionMode = forcedSlotBySlot
        ? static_cast<int>(GraphExecutionMode::GEM_SLOT_BY_SLOT) : gemMode;

    // Graph capture/replay gate. ModeContract.isSlotBySlot is the PERMANENT mode guard:
    // GEM_SLOT_BY_SLOT never captures regardless of phase (mirrors the gpubackend gate —
    // without it SLOT_BY_SLOT composite-captured the prefill and OOM-crashed once shapes
    // froze). usesGraphCapture lets an explicit graph mode (CUDA_GRAPHS/NVRTC_JIT/PTX_JIT/
    // TRITON/AUTO) capture even when tritonGraphCapture is false (BenchmarkConfigApplier
    // resets it for non-Triton configs).
    allowGraphCaptureReplay = (tritonGraphCapture || modeContract.usesGraphCapture)
                              && !modeContract.isSlotBySlot
                              && shapesFrozen && !tritonSkipKernels;

    // Variable/weight filter for performPreReplaySync H2D
    useVariableFilter = shapesFrozen && hasVariableList;

    // Triton verify flag
    tritonVerifyEnabled = tritonVerify;

    // Diagnostic gates — check if any diagnostic category is active
    diagAnyEnabled = DspDiagnostics::getInstance().isEnabled(0xFFFFFFFF);
    diagVerifyEnabled = DSP_DIAG_ENABLED(VERIFY) &&
        DspDiagnostics::getInstance().withinExecLimit(executeCount);

    // Timing
    timingEnabled = execTimingEnabled;

    // Dispatch booleans — replaces ExecutionDispatchMode enum
    // isFirstFrozenWarmup already set above.
    // forcedSlotBySlot already set above.
    // isReplay requires shapesFrozen: pre-freeze executions MUST use phaseSlotBySlot
    // to build shape caches and let AUTO_SEAL trigger. Without this gate, CUDA_GRAPHS
    // and TRITON modes enter phaseReplay on their first execution (shapesFrozen=false)
    // and hit platformShouldUseGraph() returning false → silent slot-by-slot fallback.
    isReplay = shapesFrozen && !isFirstFrozenWarmup &&
        !ModeContract::forMode(graphExecutionMode).isSlotBySlot;
  }

  /**
   * The resolved execution phase for one execute() call. ONE enum, ONE resolver
   * (resolveExecPhase) — replaces the boolean ternaries that were duplicated in
   * NativeDynamicShapePlan::execute() and here in dispatchModeName().
   */
  enum class ExecPhase { SLOT_BY_SLOT, WARMUP, REPLAY };

  /**
   * Resolve which phase this execution runs in. THE single source of truth for
   * phase dispatch — derived from the booleans populateDerivedState() already
   * computed, in priority order:
   *   1. slot-by-slot MODE (forced via skipKernels, or explicit GEM_SLOT_BY_SLOT)
   *      never captures                                               -> SLOT_BY_SLOT
   *   2. first frozen execution (isFirstFrozenWarmup) — phaseWarmup rebuilds
   *      slot/shape state exactly once                                -> WARMUP
   *   3. frozen, past warmup, a graph-replay mode (isReplay) — capture
   *      accumulates and replays here across executions               -> REPLAY
   *   4. otherwise (not yet frozen): run slot-by-slot to build shape
   *      caches and let AUTO_SEAL fire                                -> SLOT_BY_SLOT
   * graphExecutionMode is already normalized to GEM_SLOT_BY_SLOT when forcedSlotBySlot,
   * so step 1 covers both the forced and the explicit slot-by-slot cases.
   */
  SD_INLINE ExecPhase resolveExecPhase() const {
    if (ModeContract::forMode(graphExecutionMode).isSlotBySlot) return ExecPhase::SLOT_BY_SLOT;
    if (isFirstFrozenWarmup) return ExecPhase::WARMUP;
    if (isReplay)      return ExecPhase::REPLAY;
    return ExecPhase::SLOT_BY_SLOT;
  }

  SD_INLINE static const char* execPhaseName(ExecPhase p) {
    switch (p) {
      case ExecPhase::WARMUP: return "WARMUP";
      case ExecPhase::REPLAY: return "REPLAY";
      default:                return "SLOT_BY_SLOT";
    }
  }

  /**
   * Return a human-readable name for the resolved dispatch mode.
   */
  SD_INLINE const char* dispatchModeName() const {
    return execPhaseName(resolveExecPhase());
  }

  /**
   * Return a human-readable name for the current sync level.
   */
  SD_INLINE const char* syncLevelName() const {
    switch (currentSyncLevel) {
      case SyncLevel::NONE:        return "NONE";
      case SyncLevel::EVENT:       return "EVENT";
      case SyncLevel::STREAM:      return "STREAM";
      case SyncLevel::FULL_DEVICE: return "FULL_DEVICE";
      default:                     return "UNKNOWN";
    }
  }

  /**
   * Log a summary of this execution's segment dispatch and sync state.
   * Called after all segments complete, before output collection.
   */
  SD_INLINE void logExecutionSummary(int execCount) {
    if (!diagAnyEnabled) return;
    DSP_DIAG(EXECUTE,
        "EXEC_SUMMARY exec=%d mode=%s execTarget=%s sync=%s(stream=%d event=%d) syncPhase=%s "
        "segs(total=%d warmup=%d captured=%d replayed=%d sbs=%d direct=%d fail=%d invalidated=%d) "
        "frozen(const=%d shapeOnly=%d valueIndep=%d viewUnfrozen=%d javaManaged=%d) "
        "flow(%d events)",
        execCount, dispatchModeName(), execTargetName(), syncLevelName(),
        streamSyncCount, eventSyncCount, syncPhaseName(),
        segmentsTotal, segmentsWarmup, segmentsCaptured, segmentsReplayed,
        segmentsSlotBySlot, segmentsDirect, segmentsFailed, segmentsInvalidated,
        frozenConstCount, shapeOnlyTraitCount, valueIndepCount, viewAliasUnfrozen,
        javaManagedSlots, flowEventCount);
    // On failure or when diagnostics are verbose, dump the full flow log
    if (segmentsFailed > 0) {
      dumpFlowLog(execCount);
    }
  }
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_PLAN_EXECUTION_CONTEXT_H
