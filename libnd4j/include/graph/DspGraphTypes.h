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

#ifndef LIBND4J_DSP_GRAPH_TYPES_H
#define LIBND4J_DSP_GRAPH_TYPES_H

// DspGraphTypes.h — Unified "graph" terminology for the DSP execution system.
//
// The DSP system has 4 independent lifecycle dimensions (plan, segment, slot,
// replay handle) that all follow the same fundamental pattern:
//   BUILDING → SEALED → (terminal: FAILED)
//
// This header provides:
//   1. GraphNodePhase — the unified 3-state lifecycle enum that applies to ALL levels
//   2. Type aliases — new graph-centric names for existing types (zero binary impact)
//
// Usage:
//   - New code should use SubgraphNode, OpNodeDef, OpNodeState, RootGraph, etc.
//   - Old names (GraphSegment, NativeSlot, SlotEntry) remain as aliases during migration.
//   - GraphNodePhase can be queried on any level via graphNodePhase() accessors.

#include <cassert>
#include <cstdint>
// DspDiagnostics.h is included here so that the inline transition methods on
// SegmentPhase, SlotPhase, and PlanLifecycle can emit DSP_DIAG lifecycle events.
// The include guard on DspDiagnostics.h prevents double-inclusion when
// NativeDynamicShapePlan.h (which also includes DspDiagnostics.h) is the
// primary includer.
#include <graph/DspDiagnostics.h>

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// GraphNodePhase — Unified 3-state lifecycle for ALL graph nodes
// ═══════════════════════════════════════════════════════════════════════════════
//
// Every node in the execution graph (RootGraph, SubgraphNode, OpNode) progresses
// through the same lifecycle:
//
//   BUILDING ────────────────────────────► SEALED
//        │     warmup + compile + capture      │
//        │     all succeed                 steady-state
//        │                                 replay only
//        │
//        └───────────────────────────────► FAILED
//                permanent failure           slot-by-slot fallback
//
// Shape drift on a SEALED node: DESTROY the node, CREATE a new BUILDING node.
// FAILED is terminal — only plan-level invalidation resets it.
//
// Level-specific semantics:
//   RootGraph:    BUILDING = any child still building; SEALED = all children sealed
//   SubgraphNode: BUILDING = warmup/compile/capture in progress; SEALED = replay ready
//   OpNode:       BUILDING = shape inference active; SEALED = shape + buffer stable
//
// FROZEN_CONSTANT and VIEW_PRODUCER are NOT lifecycle states — they are property
// flags on a SEALED OpNode. OOM_DEFERRED is NOT a lifecycle state — it is a
// retry property on a BUILDING SubgraphNode.

enum class GraphNodePhase : uint8_t {
  BUILDING = 0,   // Construction in progress (warmup, compile, capture — all sub-phases)
  SEALED   = 1,   // Immutable steady-state (replay only, all state frozen)
  FAILED   = 2,   // Terminal failure (slot-by-slot fallback for this node)
};

// Display name for diagnostics
inline const char* graphNodePhaseName(GraphNodePhase p) {
  switch (p) {
    case GraphNodePhase::BUILDING: return "BUILDING";
    case GraphNodePhase::SEALED:   return "SEALED";
    case GraphNodePhase::FAILED:   return "FAILED";
    default:                       return "UNKNOWN";
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BuildSubPhase — progression tracker WITHIN the BUILDING state
// ═══════════════════════════════════════════════════════════════════════════════
//
// A segment in BUILDING progresses through these sub-phases in order:
//   WARMUP → COMPILING → CAPTURING → (exits to SEALED)
//
// This replaces the old 6-state SegmentLifecycleState which conflated
// lifecycle position (BUILDING vs SEALED vs FAILED) with build progression.
//
// Key simplification:
//   Old: NEEDS_WARMUP, NEEDS_COMPILE, CAPTURE_PENDING, OOM_DEFERRED, REPLAYING, FAILED
//   New: phase=BUILDING + subPhase=WARMUP/COMPILING/CAPTURING
//        phase=SEALED (was REPLAYING)
//        phase=FAILED (was FAILED)
//
// OOM_DEFERRED is a property flag (oomRetryPending), not a sub-phase.

enum class BuildSubPhase : uint8_t {
  WARMUP    = 0,  // Slot-by-slot execution to populate shape caches
  COMPILING = 1,  // Backend compilation (Triton, NVRTC, OneDNN)
  CAPTURING = 2,  // Graph capture pending (compiled, awaiting capture)
};

inline const char* buildSubPhaseName(BuildSubPhase sp) {
  switch (sp) {
    case BuildSubPhase::WARMUP:    return "WARMUP";
    case BuildSubPhase::COMPILING: return "COMPILING";
    case BuildSubPhase::CAPTURING: return "CAPTURING";
    default:                       return "UNKNOWN";
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SegmentPhase — Complete segment state in a single struct
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces:
//   - SegmentLifecycleState enum (6 states)
//   - compilationFailed bool (redundant with phase==FAILED)
//   - The mapping function graphNodePhase() that converted 6→3
//
// ONE concept, ONE check. No mapping layers.

struct SegmentPhase {
  GraphNodePhase phase = GraphNodePhase::BUILDING;
  BuildSubPhase  subPhase = BuildSubPhase::WARMUP;

  // Property flags (NOT lifecycle states)
  bool oomRetryPending = false;   // Was OOM_DEFERRED — retry scheduled
  int  oomRetryCount = 0;         // Number of OOM retries attempted
  int  oomRetryAfterExec = 0;     // Execute count at which retry fires

  // ── Phase queries ───────────────────────────────────────────────────────
  bool isBuilding()  const { return phase == GraphNodePhase::BUILDING; }
  bool isSealed()    const { return phase == GraphNodePhase::SEALED; }
  bool isFailed()    const { return phase == GraphNodePhase::FAILED; }

  bool needsWarmup()  const { return isBuilding() && subPhase == BuildSubPhase::WARMUP; }
  bool needsCompile() const { return isBuilding() && subPhase == BuildSubPhase::COMPILING; }
  bool needsCapture() const { return isBuilding() && subPhase == BuildSubPhase::CAPTURING; }

  // ── Transitions (validated) ─────────────────────────────────────────────
  // Each transition asserts the precondition. Invalid transitions are bugs.

  /** WARMUP → COMPILING (warmup passes complete) */
  void advanceToCompiling() {
    assert(isBuilding() && subPhase == BuildSubPhase::WARMUP);
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:WARMUP -> BUILDING:COMPILING");
    subPhase = BuildSubPhase::COMPILING;
  }

  /** COMPILING → CAPTURING (compilation succeeded) */
  void advanceToCapturing() {
    assert(isBuilding() && subPhase == BuildSubPhase::COMPILING);
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:COMPILING -> BUILDING:CAPTURING");
    subPhase = BuildSubPhase::CAPTURING;
  }

  /** WARMUP → CAPTURING (skip compile step — for CUDA_GRAPHS which capture inline) */
  void skipCompileToCapturing() {
    assert(isBuilding() && subPhase == BuildSubPhase::WARMUP);
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:WARMUP -> BUILDING:CAPTURING (skipped compile)");
    subPhase = BuildSubPhase::CAPTURING;
  }

  /** CAPTURING → SEALED (capture succeeded — steady-state replay) */
  void seal() {
    assert(isBuilding() && subPhase == BuildSubPhase::CAPTURING);
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:CAPTURING -> SEALED (oomRetries=%d)", oomRetryCount);
    phase = GraphNodePhase::SEALED;
    oomRetryPending = false;
    oomRetryCount = 0;
  }

  /** Any BUILDING → SEALED (non-capture terminal — for NOT_FUSIBLE segments that skip capture) */
  void sealNonCapture() {
    assert(isBuilding());
    DSP_DIAG(LIFECYCLE, "SegmentPhase: %s -> SEALED (non-capture seal)", displayName());
    phase = GraphNodePhase::SEALED;
    oomRetryPending = false;
    oomRetryCount = 0;
  }

  /** Any → FAILED (terminal, never recovers without full invalidation) */
  void fail() {
    DSP_DIAG(LIFECYCLE, "SegmentPhase: %s -> FAILED", displayName());
    phase = GraphNodePhase::FAILED;
    oomRetryPending = false;
  }

  /** Mark OOM retry (stays in CAPTURING sub-phase, sets retry flag) */
  void markOomRetry(int retryAfterExec) {
    assert(isBuilding() && subPhase == BuildSubPhase::CAPTURING);
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:CAPTURING -> BUILDING:OOM_RETRY "
             "(retryAfter=%d retryCount=%d)", retryAfterExec, oomRetryCount + 1);
    oomRetryPending = true;
    oomRetryCount++;
    oomRetryAfterExec = retryAfterExec;
  }

  /** Clear OOM flag when retry fires (still in CAPTURING) */
  void clearOomRetry() {
    DSP_DIAG(LIFECYCLE, "SegmentPhase: BUILDING:OOM_RETRY -> BUILDING:CAPTURING (retry firing)");
    oomRetryPending = false;
  }

  /** Full reset — back to initial BUILDING/WARMUP state */
  void reset() {
    DSP_DIAG(LIFECYCLE, "SegmentPhase: %s -> BUILDING:WARMUP (full reset)", displayName());
    phase = GraphNodePhase::BUILDING;
    subPhase = BuildSubPhase::WARMUP;
    oomRetryPending = false;
    oomRetryCount = 0;
    oomRetryAfterExec = 0;
  }

  // ── Display name (for diagnostics) ─────────────────────────────────────
  const char* displayName() const {
    if (isFailed()) return "FAILED";
    if (isSealed()) return "SEALED";
    if (oomRetryPending) return "BUILDING:OOM_RETRY";
    switch (subPhase) {
      case BuildSubPhase::WARMUP:    return "BUILDING:WARMUP";
      case BuildSubPhase::COMPILING: return "BUILDING:COMPILING";
      case BuildSubPhase::CAPTURING: return "BUILDING:CAPTURING";
      default:                       return "BUILDING:UNKNOWN";
    }
  }

  // ── Legacy compatibility ────────────────────────────────────────────────
  // Maps to old SegmentLifecycleState integers for JNI callers that haven't
  // migrated. Will be removed once Java side uses the new enum.
  int toLegacyCode() const {
    if (isFailed()) return 4;  // FAILED
    if (isSealed()) return 3;  // REPLAYING
    if (oomRetryPending) return 5;  // OOM_DEFERRED
    switch (subPhase) {
      case BuildSubPhase::WARMUP:    return 0;  // NEEDS_WARMUP
      case BuildSubPhase::COMPILING: return 1;  // NEEDS_COMPILE
      case BuildSubPhase::CAPTURING: return 2;  // CAPTURE_PENDING
      default: return -1;
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// SegmentExecOutcome — WHY a segment executes the way it does
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces the scattered boolean flags (captureProducedNoKernels, noFusibleOps,
// compilationFailed, oomRetryPending) with a single mutually-exclusive enum.
// SegmentPhase tracks WHERE in the lifecycle (BUILDING/SEALED/FAILED).
// SegmentExecOutcome tracks the RESULT that determines dispatch.
//
// dispatchSegment() uses (SelectedBackend, SegmentPhase, SegmentExecOutcome)
// to pick exactly one execution action.

enum class SegmentExecOutcome : uint8_t {
  PENDING          = 0,  // Still building (warmup/compile/capture in progress)
  GRAPH_REPLAY     = 1,  // Valid replay handle — steady-state graph replay
  ZERO_KERNEL_SBS  = 2,  // Capture produced 0 GPU nodes — permanent slot-by-slot
  NOT_FUSIBLE      = 3,  // Backend cannot fuse ops — permanent slot-by-slot (expected)
  COMPILE_FAILED   = 4,  // Compilation failed — permanent slot-by-slot (error)
  OOM_DEFERRED     = 5,  // OOM during capture — deferred retry scheduled
};

inline const char* segmentExecOutcomeName(SegmentExecOutcome o) {
  switch (o) {
    case SegmentExecOutcome::PENDING:         return "PENDING";
    case SegmentExecOutcome::GRAPH_REPLAY:    return "GRAPH_REPLAY";
    case SegmentExecOutcome::ZERO_KERNEL_SBS: return "ZERO_KERNEL_SBS";
    case SegmentExecOutcome::NOT_FUSIBLE:     return "NOT_FUSIBLE";
    case SegmentExecOutcome::COMPILE_FAILED:  return "COMPILE_FAILED";
    case SegmentExecOutcome::OOM_DEFERRED:    return "OOM_DEFERRED";
    default:                                  return "UNKNOWN";
  }
}

// Returns true if the outcome is terminal — segment will always execute
// slot-by-slot and never attempt graph capture again.
inline bool isTerminalOutcome(SegmentExecOutcome o) {
  return o == SegmentExecOutcome::ZERO_KERNEL_SBS ||
         o == SegmentExecOutcome::NOT_FUSIBLE ||
         o == SegmentExecOutcome::COMPILE_FAILED;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ReplayHandleTracker — tracks every mutation to a segment's replay handle
// ═══════════════════════════════════════════════════════════════════════════════
//
// Lightweight per-segment ring buffer recording CREATE, CAPTURE_BEGIN, CAPTURE_END,
// INSTANTIATE, REPLAY, INVALIDATE, DESTROY events with timestamps and counts.
// Gated behind DSP_DIAG_GRAPH_REPLAY category — zero cost when diagnostics are off.
//
// Usage from SegmentLifecycle methods: exec.handleTracker.record(...)
// Dump on demand: exec.handleTracker.dump(startSlot, endSlot)

struct ReplayHandleEvent {
  enum class Kind : uint8_t {
    CREATE          = 0,   // replay handle allocated
    CAPTURE_BEGIN   = 1,   // beginCapture() called
    CAPTURE_END     = 2,   // endCapture() returned
    INSTANTIATE     = 3,   // finalize() / cudaGraphInstantiate
    REPLAY          = 4,   // replay() / cudaGraphLaunch
    INVALIDATE      = 5,   // lifecycle invalidation (address drift, rebuild, evict)
    DESTROY         = 6,   // handle.reset() — cudaGraphExecDestroy
    ADDRESS_DRIFT   = 7,   // address mismatch detected before replay
    OOM_DEFERRED    = 8,   // instantiate failed with OOM
    EXEC_ERROR      = 9,   // any error (capture fail, replay fail)
  };

  Kind     kind;
  int      execCount;        // plan execution count at event time
  int64_t  timestampUs;      // microseconds since epoch (steady_clock)
  int      numNodes;         // graph node count (for CAPTURE_END/INSTANTIATE), 0 otherwise
  int      cudaError;        // CUDA error code (for ERROR), 0 otherwise
  char     reason[64];       // short reason string (truncated)
};

inline const char* replayHandleEventKindName(ReplayHandleEvent::Kind k) {
  switch (k) {
    case ReplayHandleEvent::Kind::CREATE:        return "CREATE";
    case ReplayHandleEvent::Kind::CAPTURE_BEGIN:  return "CAPTURE_BEGIN";
    case ReplayHandleEvent::Kind::CAPTURE_END:    return "CAPTURE_END";
    case ReplayHandleEvent::Kind::INSTANTIATE:    return "INSTANTIATE";
    case ReplayHandleEvent::Kind::REPLAY:         return "REPLAY";
    case ReplayHandleEvent::Kind::INVALIDATE:     return "INVALIDATE";
    case ReplayHandleEvent::Kind::DESTROY:        return "DESTROY";
    case ReplayHandleEvent::Kind::ADDRESS_DRIFT:  return "ADDRESS_DRIFT";
    case ReplayHandleEvent::Kind::OOM_DEFERRED:   return "OOM_DEFERRED";
    case ReplayHandleEvent::Kind::EXEC_ERROR:     return "ERROR";
    default:                                      return "UNKNOWN";
  }
}

class ReplayHandleTracker {
 public:
  static constexpr int RING_SIZE = 128;
  static constexpr int RING_MASK = RING_SIZE - 1;

  ReplayHandleTracker() { reset(); }

  void record(ReplayHandleEvent::Kind kind, int execCount,
              int numNodes = 0, int cudaError = 0,
              const char* reason = nullptr) {
    int idx = writePos_++ & RING_MASK;
    auto& e = events_[idx];
    e.kind = kind;
    e.execCount = execCount;
    e.timestampUs = nowUs();
    e.numNodes = numNodes;
    e.cudaError = cudaError;
    if (reason) {
      std::strncpy(e.reason, reason, sizeof(e.reason) - 1);
      e.reason[sizeof(e.reason) - 1] = '\0';
    } else {
      e.reason[0] = '\0';
    }
    // Update aggregate counters
    switch (kind) {
      case ReplayHandleEvent::Kind::CREATE:        createCount_++; break;
      case ReplayHandleEvent::Kind::CAPTURE_BEGIN:  captureBeginCount_++; break;
      case ReplayHandleEvent::Kind::CAPTURE_END:    captureEndCount_++; break;
      case ReplayHandleEvent::Kind::INSTANTIATE:    instantiateCount_++; break;
      case ReplayHandleEvent::Kind::REPLAY:         replayCount_++; break;
      case ReplayHandleEvent::Kind::INVALIDATE:     invalidateCount_++; break;
      case ReplayHandleEvent::Kind::DESTROY:        destroyCount_++; break;
      case ReplayHandleEvent::Kind::ADDRESS_DRIFT:  addressDriftCount_++; break;
      case ReplayHandleEvent::Kind::OOM_DEFERRED:   oomCount_++; break;
      case ReplayHandleEvent::Kind::EXEC_ERROR:     errorCount_++; break;
    }
  }

  // Dump recent events via DspDiagnostics
  void dump(int startSlot, int endSlot, int lastN = 32) const {
    int total = writePos_;
    int start = (total > lastN) ? (total - lastN) : 0;
    sd_printf("=== REPLAY HANDLE TRACKER: seg[%d-%d] === "
              "creates=%d captures=%d instantiates=%d replays=%d "
              "invalidates=%d destroys=%d drifts=%d ooms=%d errors=%d\n",
              startSlot, endSlot,
              createCount_, captureEndCount_, instantiateCount_, replayCount_,
              invalidateCount_, destroyCount_, addressDriftCount_, oomCount_, errorCount_);
    for (int i = start; i < total; i++) {
      const auto& e = events_[i & RING_MASK];
      sd_printf("  [%d] %s exec=%d nodes=%d cudaErr=%d reason='%s'\n",
                i, replayHandleEventKindName(e.kind),
                e.execCount, e.numNodes, e.cudaError, e.reason);
    }
  }

  // JSON-serializable summary for diagnostic reports
  std::string toJsonSummary(int startSlot, int endSlot) const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"seg\":[%d,%d],\"creates\":%d,\"captureBegins\":%d,"
        "\"captureEnds\":%d,\"instantiates\":%d,\"replays\":%d,"
        "\"invalidates\":%d,\"destroys\":%d,\"drifts\":%d,"
        "\"ooms\":%d,\"errors\":%d,\"totalEvents\":%d}",
        startSlot, endSlot,
        createCount_, captureBeginCount_, captureEndCount_,
        instantiateCount_, replayCount_, invalidateCount_,
        destroyCount_, addressDriftCount_, oomCount_, errorCount_,
        writePos_);
    return std::string(buf);
  }

  // Aggregate accessors
  int createCount() const { return createCount_; }
  int captureBeginCount() const { return captureBeginCount_; }
  int captureEndCount() const { return captureEndCount_; }
  int instantiateCount() const { return instantiateCount_; }
  int replayCount() const { return replayCount_; }
  int invalidateCount() const { return invalidateCount_; }
  int destroyCount() const { return destroyCount_; }
  int addressDriftCount() const { return addressDriftCount_; }
  int oomCount() const { return oomCount_; }
  int errorCount() const { return errorCount_; }
  int totalEvents() const { return writePos_; }

  void reset() {
    writePos_ = 0;
    createCount_ = captureBeginCount_ = captureEndCount_ = 0;
    instantiateCount_ = replayCount_ = invalidateCount_ = 0;
    destroyCount_ = addressDriftCount_ = oomCount_ = errorCount_ = 0;
    std::memset(events_, 0, sizeof(events_));
  }

 private:
  static int64_t nowUs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
  }

  ReplayHandleEvent events_[RING_SIZE];
  int writePos_ = 0;
  int createCount_ = 0;
  int captureBeginCount_ = 0;
  int captureEndCount_ = 0;
  int instantiateCount_ = 0;
  int replayCount_ = 0;
  int invalidateCount_ = 0;
  int destroyCount_ = 0;
  int addressDriftCount_ = 0;
  int oomCount_ = 0;
  int errorCount_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// SlotPhase — Complete slot state in a single struct
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces:
//   - SlotLifecycleState enum (4 states: WARMUP, SHAPE_CACHED, FROZEN, FROZEN_CONSTANT)
//   - The isConstant() / isViewProducer() checks scattered across code
//
// Design: GraphNodePhase + property flags.
//   BUILDING = shape not yet stable (warmup or shape-cached but not frozen)
//   SEALED   = shape + buffer address stable (ready for graph capture)
//   FAILED   = not used for slots currently (all slots eventually stabilize)
//
// Property flags (on SEALED):
//   isConstant     — slot output never changes (weights, shape-only ops)
//   isViewProducer — slot output is a view of another slot's buffer

struct SlotPhase {
  GraphNodePhase phase = GraphNodePhase::BUILDING;

  // Property flags (meaningful when SEALED)
  bool isConstant = false;       // Was FROZEN_CONSTANT
  bool isViewProducer = false;   // Output is a view (reshape, permute, etc.)
  bool shapeCacheValid = false;  // Shape has been observed at least once

  // ── Phase queries ───────────────────────────────────────────────────────
  bool isBuilding() const { return phase == GraphNodePhase::BUILDING; }
  bool isSealed()   const { return phase == GraphNodePhase::SEALED; }
  bool isFrozen()   const { return isSealed(); }  // Legacy name

  // ── Transitions ─────────────────────────────────────────────────────────

  /** Mark shape as observed (stays BUILDING but caches shape info) */
  void markShapeCached() {
    DSP_DIAG(LIFECYCLE, "SlotPhase: %s -> BUILDING:SHAPE_CACHED", displayName());
    shapeCacheValid = true;
  }

  /** Freeze slot (shape + buffer stable → SEALED)
   *
   * isViewProducer is a STRUCTURAL property of the op (permute/reshape always
   * produce views) — not an execution-phase property. seal() therefore only
   * SETS isViewProducer (when viewProducer=true), and never clears it.
   * This preserves the flag that was set by the view-op execution path
   * (slots_[slotIdx].slotPhase.isViewProducer = true) so that subsequent calls
   * to refreshStaleViewWrappersInSegment correctly detect and refresh stale
   * views after placeholder inputs are closed and re-opened between iterations.
   *
   * Regression: prior to this fix, seal(false) reset isViewProducer to false,
   * causing refreshStaleViewWrappersInSegment to skip the view slot and leave
   * a dangling NDArray pointing at a freed DataBuffer. */
  void seal(bool constant = false, bool viewProducer = false) {
    DSP_DIAG(LIFECYCLE, "SlotPhase: %s -> SEALED (constant=%d viewProducer=%d existing=%d)",
             displayName(), (int)constant, (int)viewProducer, (int)isViewProducer);
    phase = GraphNodePhase::SEALED;
    isConstant = constant;
    // Preserve isViewProducer: only set it, never clear it via seal().
    // The view-op execution path sets it to true before seal() fires;
    // clearing it here causes refreshStaleViewWrappersInSegment to skip
    // view slots and leave dangling pointers after placeholder close/reuse.
    if (viewProducer) isViewProducer = true;
  }

  /** Unfreeze (back to BUILDING — e.g., shape change detected) */
  void unseal() {
    DSP_DIAG(LIFECYCLE, "SlotPhase: %s -> BUILDING:WARMUP (unsealed — shape change, viewProducer=%d preserved)",
             displayName(), (int)isViewProducer);
    phase = GraphNodePhase::BUILDING;
    isConstant = false;
    // NOTE: isViewProducer is NOT cleared here. It is a structural property
    // of the op (permute/reshape always produce views) — not an execution-phase
    // property. Clearing it causes refreshStaleViewWrappersInSegment to skip
    // the slot, leaving a dangling view whose DataBuffer has been freed.
    shapeCacheValid = false;
  }

  /** Full reset */
  void reset() {
    DSP_DIAG(LIFECYCLE, "SlotPhase: %s -> BUILDING:WARMUP (full reset, viewProducer=%d preserved)",
             displayName(), (int)isViewProducer);
    phase = GraphNodePhase::BUILDING;
    isConstant = false;
    // NOTE: isViewProducer is NOT cleared here — same rationale as unseal().
    // The flag is set once when the op first executes and must persist across
    // phase transitions so that refreshStaleViewWrappersInSegment can detect
    // and refresh stale views after placeholder inputs are closed.
    shapeCacheValid = false;
  }

  // ── Display name ────────────────────────────────────────────────────────
  const char* displayName() const {
    if (isSealed() && isConstant) return "SEALED:CONSTANT";
    if (isSealed() && isViewProducer) return "SEALED:VIEW";
    if (isSealed()) return "SEALED";
    if (shapeCacheValid) return "BUILDING:SHAPE_CACHED";
    return "BUILDING:WARMUP";
  }

  // ── Legacy compatibility ────────────────────────────────────────────────
  int toLegacyCode() const {
    if (isSealed() && isConstant) return 3;  // FROZEN_CONSTANT
    if (isSealed()) return 2;                // FROZEN
    if (shapeCacheValid) return 1;           // SHAPE_CACHED
    return 0;                                // WARMUP
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// PlanLifecycle — unified plan-level lifecycle struct
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces the scattered combination of planPhase_ (enum), shapesFrozen_ (bool),
// pointersStable_, pointersStableCount_, and postFreezeExecCount_.
//
// Single struct = single source of truth. Impossible to have shapesFrozen=true
// while phase=SLOT_BY_SLOT. The struct enforces invariants via transitions.
//
// Lifecycle:
//   SLOT_BY_SLOT (BUILDING) → SHAPES_FROZEN (BUILDING) → REPLAYING (SEALED)
//   Any state can → FAILED (terminal without full reset)

struct PlanLifecycle {
  GraphNodePhase phase = GraphNodePhase::BUILDING;

  // Sub-phase within BUILDING (mirrors the old PlanPhase enum values)
  enum class BuildStage : uint8_t {
    SLOT_BY_SLOT = 0,     // No guarantees — shapes and pointers may change
    SHAPES_FROZEN = 1,    // Shapes constant, tracking pointer stability
    REPLAY_BLOCKED = 2,   // Frozen but zero GRAPH_REPLAY segments — will never enter REPLAYING
  };

  BuildStage buildStage = BuildStage::SLOT_BY_SLOT;

  // Pointer stability tracking (meaningful only in SHAPES_FROZEN)
  int  pointersStableCount = 0;   // Consecutive steps with stable arg tables
  int  postFreezeExecCount = 0;   // Executions since shapes froze

  // Property flags
  bool compilationDone = false;   // Precompilation step completed

  // ── Phase queries ───────────────────────────────────────────────────────
  bool isBuilding()  const { return phase == GraphNodePhase::BUILDING; }
  bool isSealed()    const { return phase == GraphNodePhase::SEALED; }
  bool isFailed()    const { return phase == GraphNodePhase::FAILED; }

  bool isSlotBySlot()    const { return isBuilding() && buildStage == BuildStage::SLOT_BY_SLOT; }
  bool isShapesFrozen()  const { return isBuilding() && buildStage == BuildStage::SHAPES_FROZEN; }
  bool isReplayBlocked() const { return isBuilding() && buildStage == BuildStage::REPLAY_BLOCKED; }
  bool isReplaying()     const { return isSealed(); }

  /** True when shapes are stable — either SHAPES_FROZEN (building) or REPLAYING (sealed).
   *  This is the correct check for "should we use frozen/replay code paths?" because
   *  isShapesFrozen() returns FALSE once the plan transitions to REPLAYING (SEALED).
   *  Using isShapesFrozen() alone causes the plan to fall back to slot-by-slot after
   *  the SHAPES_FROZEN → REPLAYING transition, breaking CUDA_GRAPHS replay. */
  bool isInFrozenOrReplayState() const { return isShapesFrozen() || isReplaying(); }

  bool pointersStable()  const { return isSealed() || pointersStableCount >= 2; }

  // ── Transitions ─────────────────────────────────────────────────────────

  /** SLOT_BY_SLOT → SHAPES_FROZEN (shapes observed constant) */
  void freezeShapes() {
    assert(isBuilding() && buildStage == BuildStage::SLOT_BY_SLOT);
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: SLOT_BY_SLOT -> SHAPES_FROZEN");
    buildStage = BuildStage::SHAPES_FROZEN;
    postFreezeExecCount = 0;
    pointersStableCount = 0;
  }

  /** SHAPES_FROZEN → REPLAYING (pointers stable for 2+ steps) */
  void seal() {
    assert(isBuilding() && buildStage == BuildStage::SHAPES_FROZEN);
    assert(pointersStableCount >= 2);
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: SHAPES_FROZEN -> REPLAYING "
             "(pointersStableCount=%d postFreezeExec=%d)",
             pointersStableCount, postFreezeExecCount);
    phase = GraphNodePhase::SEALED;
  }

  /** SHAPES_FROZEN → REPLAY_BLOCKED (zero GRAPH_REPLAY segments — cannot enter REPLAYING) */
  void blockReplay(const char* reason) {
    assert(isBuilding() && buildStage == BuildStage::SHAPES_FROZEN);
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: SHAPES_FROZEN -> REPLAY_BLOCKED reason=%s",
             reason ? reason : "?");
    buildStage = BuildStage::REPLAY_BLOCKED;
  }

  /** Unfreeze — back to SLOT_BY_SLOT (shape change detected) */
  void unfreeze() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: %s -> SLOT_BY_SLOT (shape change detected)", displayName());
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SLOT_BY_SLOT;
    pointersStableCount = 0;
    postFreezeExecCount = 0;
  }

  /** REPLAYING → SHAPES_FROZEN (pointer drift detected — need re-capture) */
  void unseal() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: REPLAYING -> SHAPES_FROZEN (pointer drift — need re-capture)");
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SHAPES_FROZEN;
    pointersStableCount = 0;
  }

  /** Any → FAILED */
  void fail() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: %s -> FAILED", displayName());
    phase = GraphNodePhase::FAILED;
  }

  /** Full reset */
  void reset() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: %s -> SLOT_BY_SLOT (full reset)", displayName());
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SLOT_BY_SLOT;
    pointersStableCount = 0;
    postFreezeExecCount = 0;
    compilationDone = false;
  }

  /** Record one stable-pointers observation */
  void recordPointersStable() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: pointersStableCount %d -> %d (stable observation)",
             pointersStableCount, pointersStableCount + 1);
    pointersStableCount++;
  }

  /** Record one unstable-pointers observation (resets counter) */
  void recordPointersUnstable() {
    DSP_DIAG(LIFECYCLE, "PlanLifecycle: pointersStableCount %d -> 0 (unstable — pointers drifted)",
             pointersStableCount);
    pointersStableCount = 0;
  }

  /** Increment post-freeze execution count */
  void recordPostFreezeExec() {
    postFreezeExecCount++;
  }

  // ── Display name ────────────────────────────────────────────────────────
  const char* displayName() const {
    if (isFailed()) return "FAILED";
    if (isSealed()) return "REPLAYING";
    if (isReplayBlocked()) return "REPLAY_BLOCKED";
    if (isShapesFrozen()) return "SHAPES_FROZEN";
    return "SLOT_BY_SLOT";
  }

  // ── Legacy compatibility ────────────────────────────────────────────────
  int toLegacyCode() const {
    if (isSealed()) return 2;           // REPLAYING
    if (isReplayBlocked()) return 3;    // REPLAY_BLOCKED
    if (isShapesFrozen()) return 1;     // SHAPES_FROZEN
    return 0;                            // SLOT_BY_SLOT
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Forward declarations for type aliases
// (actual definitions in NativeDynamicShapePlan.h, SlotArray.h, etc.)
// ═══════════════════════════════════════════════════════════════════════════════

// These forward-declare the OLD names so that the using declarations below
// are valid even if this header is included before the full definitions.
struct GraphSegment;
struct GraphSegmentDef;
struct GraphSegmentExec;
struct NativeSlot;
struct SlotEntry;
class SlotArray;
class NativeDynamicShapePlan;
class PlanTopology;
class PlanDefinition;
class SegmentExecutor;
enum class SlotLifecycleState : uint8_t;

// ═══════════════════════════════════════════════════════════════════════════════
// Type Aliases — Graph-centric terminology
// ═══════════════════════════════════════════════════════════════════════════════
//
// New code should use these names. Old names remain fully functional.
// Once all call sites migrate (Phase 5), the old names become the aliases.
//
// Hierarchy:
//   RootGraph (top-level plan)
//     └── SubgraphNode (capturable segment — contains a range of ops)
//           └── OpNodeDef (per-op immutable definition)
//           └── OpNodeState (per-op mutable state: array, lifecycle, generation)
//
// Note: using declarations with forward-declared types are valid in C++11+.
// They become full aliases once the definition is visible in the TU.

using SubgraphNode      = GraphSegment;
using SubgraphDef       = GraphSegmentDef;
using SubgraphExec      = GraphSegmentExec;
using SubgraphExecutor  = SegmentExecutor;
using OpNodeDef         = NativeSlot;
using OpNodeState       = SlotEntry;
using OpNodeStateArray  = SlotArray;
using RootGraph         = NativeDynamicShapePlan;
using RootGraphTopology = PlanTopology;
using RootGraphDef      = PlanDefinition;
using OpNodeLifecycle   = SlotLifecycleState;

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_GRAPH_TYPES_H
