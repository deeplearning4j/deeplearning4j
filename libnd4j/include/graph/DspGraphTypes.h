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
    subPhase = BuildSubPhase::COMPILING;
  }

  /** COMPILING → CAPTURING (compilation succeeded) */
  void advanceToCapturing() {
    assert(isBuilding() && subPhase == BuildSubPhase::COMPILING);
    subPhase = BuildSubPhase::CAPTURING;
  }

  /** WARMUP → CAPTURING (skip compile step — for CUDA_GRAPHS which capture inline) */
  void skipCompileToCapturing() {
    assert(isBuilding() && subPhase == BuildSubPhase::WARMUP);
    subPhase = BuildSubPhase::CAPTURING;
  }

  /** CAPTURING → SEALED (capture succeeded — steady-state replay) */
  void seal() {
    assert(isBuilding() && subPhase == BuildSubPhase::CAPTURING);
    phase = GraphNodePhase::SEALED;
    oomRetryPending = false;
    oomRetryCount = 0;
  }

  /** Any → FAILED (terminal, never recovers without full invalidation) */
  void fail() {
    phase = GraphNodePhase::FAILED;
    oomRetryPending = false;
  }

  /** Mark OOM retry (stays in CAPTURING sub-phase, sets retry flag) */
  void markOomRetry(int retryAfterExec) {
    assert(isBuilding() && subPhase == BuildSubPhase::CAPTURING);
    oomRetryPending = true;
    oomRetryCount++;
    oomRetryAfterExec = retryAfterExec;
  }

  /** Clear OOM flag when retry fires (still in CAPTURING) */
  void clearOomRetry() {
    oomRetryPending = false;
  }

  /** Full reset — back to initial BUILDING/WARMUP state */
  void reset() {
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
    shapeCacheValid = true;
  }

  /** Freeze slot (shape + buffer stable → SEALED) */
  void seal(bool constant = false, bool viewProducer = false) {
    phase = GraphNodePhase::SEALED;
    isConstant = constant;
    isViewProducer = viewProducer;
  }

  /** Unfreeze (back to BUILDING — e.g., shape change detected) */
  void unseal() {
    phase = GraphNodePhase::BUILDING;
    isConstant = false;
    isViewProducer = false;
    shapeCacheValid = false;
  }

  /** Full reset */
  void reset() {
    phase = GraphNodePhase::BUILDING;
    isConstant = false;
    isViewProducer = false;
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
  bool isReplaying()     const { return isSealed(); }

  bool pointersStable()  const { return isSealed() || pointersStableCount >= 2; }

  // ── Transitions ─────────────────────────────────────────────────────────

  /** SLOT_BY_SLOT → SHAPES_FROZEN (shapes observed constant) */
  void freezeShapes() {
    assert(isBuilding() && buildStage == BuildStage::SLOT_BY_SLOT);
    buildStage = BuildStage::SHAPES_FROZEN;
    postFreezeExecCount = 0;
    pointersStableCount = 0;
  }

  /** SHAPES_FROZEN → REPLAYING (pointers stable for 2+ steps) */
  void seal() {
    assert(isBuilding() && buildStage == BuildStage::SHAPES_FROZEN);
    assert(pointersStableCount >= 2);
    phase = GraphNodePhase::SEALED;
  }

  /** Unfreeze — back to SLOT_BY_SLOT (shape change detected) */
  void unfreeze() {
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SLOT_BY_SLOT;
    pointersStableCount = 0;
    postFreezeExecCount = 0;
  }

  /** REPLAYING → SHAPES_FROZEN (pointer drift detected — need re-capture) */
  void unseal() {
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SHAPES_FROZEN;
    pointersStableCount = 0;
  }

  /** Any → FAILED */
  void fail() {
    phase = GraphNodePhase::FAILED;
  }

  /** Full reset */
  void reset() {
    phase = GraphNodePhase::BUILDING;
    buildStage = BuildStage::SLOT_BY_SLOT;
    pointersStableCount = 0;
    postFreezeExecCount = 0;
    compilationDone = false;
  }

  /** Record one stable-pointers observation */
  void recordPointersStable() {
    pointersStableCount++;
  }

  /** Record one unstable-pointers observation (resets counter) */
  void recordPointersUnstable() {
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
    if (isShapesFrozen()) return "SHAPES_FROZEN";
    return "SLOT_BY_SLOT";
  }

  // ── Legacy compatibility ────────────────────────────────────────────────
  int toLegacyCode() const {
    if (isSealed()) return 2;           // REPLAYING
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
