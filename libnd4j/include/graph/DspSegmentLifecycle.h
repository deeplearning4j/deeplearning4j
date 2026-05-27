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

#pragma once

// SegmentLifecycle — state machine for DSP segment execution phases.
//
// This is shared between NativeDynamicShapePlan_gpubackend.cpp (platform-agnostic
// dispatch) and NativeDynamicShapePlan_gpubackend.cu (CUDA-specific execution).
// All functions are static inline to avoid ODR violations across TUs.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <cassert>

namespace sd {
namespace graph {
namespace SegmentLifecycle {

using SLS = GraphSegmentExec::SegmentLifecycleState;

static inline const char* stateName(SLS s) {
  switch (s) {
    case SLS::NEEDS_WARMUP:    return "NEEDS_WARMUP";
    case SLS::NEEDS_COMPILE:   return "NEEDS_COMPILE";
    case SLS::CAPTURE_PENDING: return "CAPTURE_PENDING";
    case SLS::REPLAYING:       return "REPLAYING";
    case SLS::FAILED:          return "FAILED";
    case SLS::OOM_DEFERRED:    return "OOM_DEFERRED";
    default:                   return "UNKNOWN";
  }
}

// ── Transition validation ────────────────────────────────────────────────
//
// Valid transitions:
//   NEEDS_WARMUP    → NEEDS_COMPILE     (markWarmupDone)
//   NEEDS_COMPILE   → CAPTURE_PENDING   (markCompiled)
//   CAPTURE_PENDING → REPLAYING         (markCaptured)  — one capture, many replays
//   CAPTURE_PENDING → OOM_DEFERRED      (markOomDeferred)
//   any             → FAILED            (markFailed)     — terminal
//   any             → NEEDS_WARMUP      (invalidateForRebuild) — full reset
//
// The FAILED state is terminal: only invalidateForRebuild can leave it.
// REPLAYING is steady-state: only invalidateForRebuild can leave it.

#ifndef __CUDA_ARCH__
// SLS_ASSERT_FROM: validates segPhase (source of truth) matches expected old SLS state.
// Mapping: NEEDS_WARMUP→needsWarmup(), NEEDS_COMPILE→needsCompile(),
//          CAPTURE_PENDING→needsCapture(), REPLAYING→isSealed(), FAILED→isFailed()
static inline bool segPhaseMatchesSLS(const SegmentPhase& sp, SLS expected) {
  switch (expected) {
    case SLS::NEEDS_WARMUP:    return sp.needsWarmup();
    case SLS::NEEDS_COMPILE:   return sp.needsCompile();
    case SLS::CAPTURE_PENDING: return sp.needsCapture();
    case SLS::REPLAYING:       return sp.isSealed();
    case SLS::FAILED:          return sp.isFailed();
    case SLS::OOM_DEFERRED:    return sp.needsCapture() && sp.oomRetryPending;
    default:                   return false;
  }
}

#define SLS_ASSERT_FROM(exec, expected, targetName) \
  do { \
    if (!segPhaseMatchesSLS(exec.segPhase, (expected))) { \
      sd_printf("DSP LIFECYCLE VIOLATION: %s requires state %s, but segment is in %s\n", \
                (targetName), stateName(expected), exec.segPhase.displayName()); \
      DSP_DIAG(FALLBACK, "LIFECYCLE_VIOLATION: %s requires %s, actual %s", \
               (targetName), stateName(expected), exec.segPhase.displayName()); \
      assert(false && "DSP segment lifecycle violation"); \
    } \
  } while (0)

#define SLS_ASSERT_NOT_TERMINAL(exec, targetName) \
  do { \
    if (exec.segPhase.isFailed()) { \
      sd_printf("DSP LIFECYCLE VIOLATION: %s called on FAILED segment " \
                "(segPhase=%s, use invalidateForRebuild to reset)\n", \
                (targetName), exec.segPhase.displayName()); \
      DSP_DIAG(FALLBACK, "LIFECYCLE_VIOLATION: %s on FAILED segment (segPhase=%s)", \
               (targetName), exec.segPhase.displayName()); \
      assert(false && "DSP transition from FAILED without invalidation"); \
    } \
  } while (0)
#else
#define SLS_ASSERT_FROM(exec, expected, targetName) ((void)0)
#define SLS_ASSERT_NOT_TERMINAL(exec, targetName) ((void)0)
#endif

// BUILDING:WARMUP -> BUILDING:COMPILING
static inline void markWarmupDone(GraphSegmentExec& exec) {
  SLS_ASSERT_FROM(exec, SLS::NEEDS_WARMUP, "markWarmupDone");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:COMPILING", exec.segPhase.displayName());
  exec.segPhase.advanceToCompiling();  // PRIMARY
  exec.lifecycleState = SLS::NEEDS_COMPILE;  // Legacy sync
}

// BUILDING:COMPILING -> BUILDING:CAPTURING
static inline void markCompiled(GraphSegmentExec& exec, const char* backendName, LongType shapeKey) {
  SLS_ASSERT_FROM(exec, SLS::NEEDS_COMPILE, "markCompiled");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:CAPTURING (backend=%s shapeKey=%lld execCount=%d)",
           exec.segPhase.displayName(), backendName, (long long)shapeKey, exec.executionCount);
  exec.segPhase.advanceToCapturing();  // PRIMARY
  exec.compiledByBackend = backendName;
  exec.lifecycleState = SLS::CAPTURE_PENDING;  // Legacy sync
}

// BUILDING:CAPTURING -> SEALED (capture complete — steady-state replay)
static inline void markCaptured(GraphSegmentExec& exec,
                                LongType inputAddrKey, LongType createValueKey,
                                LongType slotAddrHash, const char* backendName) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markCaptured");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> SEALED (backend=%s inputAddrKey=%lld "
           "createValueKey=%lld slotAddrHash=%lld execCount=%d needsArgRefresh=%d)",
           exec.segPhase.displayName(), backendName,
           (long long)inputAddrKey, (long long)createValueKey,
           (long long)slotAddrHash, exec.executionCount, (int)exec.needsArgRefresh());
  exec.segPhase.seal();  // PRIMARY: BUILDING:CAPTURING → SEALED
  exec.outcome = SegmentExecOutcome::GRAPH_REPLAY;
  exec.capturedInputAddrKey = inputAddrKey;
  exec.capturedCreateValueKey = createValueKey;
  exec.capturedSlotAddrHash = slotAddrHash;
  exec.gapOpsCapturedInGraph = false;
  if (exec.compiledByBackend.empty()) exec.compiledByBackend = backendName;
  exec.lifecycleState = SLS::REPLAYING;  // Legacy sync
}

// Legacy compatibility — markReplaying is now a no-op since markCaptured
// transitions directly to SEALED. Callers that previously did
// CAPTURED→REPLAYING on first replay are harmless no-ops.
static inline void markReplaying(GraphSegmentExec& exec) {
  (void)exec;
}

// any -> FAILED (terminal)
static inline void markFailed(GraphSegmentExec& exec, const char* reason) {
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> FAILED (reason=%s execCount=%d compiledBy=%s)",
           exec.segPhase.displayName(), reason, exec.executionCount,
           exec.compiledByBackend.c_str());
  exec.segPhase.fail();  // PRIMARY
  exec.compilationFailed = true;
  exec.outcome = SegmentExecOutcome::COMPILE_FAILED;
  exec.lifecycleState = SLS::FAILED;  // Legacy sync
}

// BUILDING:CAPTURING -> BUILDING:OOM_RETRY
static inline void markOomDeferred(GraphSegmentExec& exec, int retryAfterExec) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markOomDeferred");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:OOM_RETRY (retryAfter=%d retries=%d)",
           exec.segPhase.displayName(), retryAfterExec, exec.captureOomRetries + 1);
  exec.segPhase.markOomRetry(retryAfterExec);  // PRIMARY
  exec.captureOomRetries++;
  exec.captureRetryAfterExec = retryAfterExec;
  exec.outcome = SegmentExecOutcome::OOM_DEFERRED;
  exec.lifecycleState = SLS::OOM_DEFERRED;  // Legacy sync
}

// any -> NEEDS_WARMUP (segment-only invalidation — no plan-level resetExecuteCount)
// Use when segment captures (CUDA graphs) are stale but plan shapes haven't changed.
// This avoids resetting executeCount_ which would trigger destructive phaseWarmup.
static inline void invalidateSegmentCaptures(NativeDynamicShapePlan* plan, GraphSegment& seg,
                                              const char* reason) {
  auto& exec = seg.exec;
  DSP_DIAG(EXECUTE, "invalidateSegmentCaptures: seg[%d-%d] from=%s reason=%s "
           "execCount=%d needsArgRefresh=%d compiledBy=%s",
           seg.def.startSlot, seg.def.endSlot, exec.segPhase.displayName(), reason,
           exec.executionCount, (int)exec.needsArgRefresh(), exec.compiledByBackend.c_str());
  DSP_SEG_EVENT(seg, INVALIDATE, "capturesOnly reason=%s", reason);
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (invalidateCaptures: %s)",
           exec.segPhase.displayName(), reason);
  plan->cleanupSegmentForRebuild(seg, reason);
  plan->clearNativeRangeSegmentsForSlotRange(seg.def.startSlot, seg.def.endSlot);

  // Reset per-slot slotPhase for all slots in this segment.
  // Without this, slots remain SEALED after invalidation and executeSlot
  // enters the frozen context path with stale cached inputs/outputs —
  // causing the first post-invalidation execution to produce the same
  // result as the last pre-invalidation execution (step 0 = step 1 bug).
  plan->resetSlotStatesForSegment(seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "invalidateSegmentCaptures: reset slotPhase for slots [%d-%d] "
           "back to BUILDING (reason=%s)",
           seg.def.startSlot, seg.def.endSlot, reason);

  exec.segPhase.reset();  // PRIMARY: back to BUILDING:WARMUP
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.cachedShapeKey = 0;
  exec.capturedInputAddrKey = 0;
  exec.capturedCreateValueKey = 0;
  exec.capturedSlotAddrHash = 0;
  exec.compilationFailed = false;
  exec.bumpArgGeneration();
  exec.addrKeyStableCount = 0;
  exec.slotAddrStableCount = 0;
  exec.compiledByBackend.clear();
  exec.executionCount = 0;
  exec.lastReplayExecCount = 0;
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
  // NOTE: intentionally do NOT call plan->resetExecuteCount() here.
  // The plan shapes are unchanged — only segment CUDA graph captures are stale.
  // Resetting executeCount would cause isFirstFrozenWarmup=true → phaseWarmup
  // fires destructively, destroying all CUDA graphs and producing zeros.
}

// any -> NEEDS_WARMUP (full invalidation)
static inline void invalidateForRebuild(NativeDynamicShapePlan* plan, GraphSegment& seg,
                                        const char* reason) {
  auto& exec = seg.exec;
  DSP_DIAG(EXECUTE, "invalidateForRebuild: seg[%d-%d] from=%s reason=%s "
           "execCount=%d needsArgRefresh=%d compiledBy=%s — will resetExecuteCount+resetFrozenConstant",
           seg.def.startSlot, seg.def.endSlot, exec.segPhase.displayName(), reason,
           exec.executionCount, (int)exec.needsArgRefresh(), exec.compiledByBackend.c_str());
  DSP_SEG_EVENT(seg, INVALIDATE, "reason=%s", reason);
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (invalidate: %s)",
           exec.segPhase.displayName(), reason);
  plan->cleanupSegmentForRebuild(seg, reason);
  // Clear any nativeRangeSegments_ entries within this segment's slot range.
  // These hold FunctionalReplayHandle captures that reference the OLD slot array
  // state. If they persist, the NativeSlotExecutor lambda replays stale data on
  // the next token instead of re-capturing against the rebuilt slot state.
  plan->clearNativeRangeSegmentsForSlotRange(seg.def.startSlot, seg.def.endSlot);

  // Reset per-slot slotPhase for all slots in this segment.
  // Without this, slots remain SEALED after invalidation and executeSlot
  // enters the frozen context path with stale cached inputs/outputs —
  // same stale-frozen bug as invalidateSegmentCaptures. Full rebuild must
  // reset slot states too, otherwise the post-invalidation warmup executes
  // stale contexts for the first step.
  plan->resetSlotStatesForSegment(seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "invalidateForRebuild: reset slotPhase for slots [%d-%d] "
           "back to BUILDING (reason=%s)",
           seg.def.startSlot, seg.def.endSlot, reason);

  exec.segPhase.reset();  // PRIMARY: back to BUILDING:WARMUP
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.cachedShapeKey = 0;
  exec.capturedInputAddrKey = 0;
  exec.capturedCreateValueKey = 0;
  exec.capturedSlotAddrHash = 0;
  exec.compilationFailed = false;
  exec.bumpArgGeneration();
  exec.addrKeyStableCount = 0;
  exec.slotAddrStableCount = 0;
  exec.compiledByBackend.clear();
  exec.executionCount = 0;
  exec.lastReplayExecCount = 0;
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
  // Reset plan-level executeCount so isFirstFrozenWarmup evaluates to true
  // on the next execute().  Without this, executeCount_ stays high after
  // invalidation, isFirstFrozenWarmup = (shapesFrozen && executeCount==0)
  // is false, and the plan skips phaseWarmup — going straight to
  // phaseReplay/phaseSlotBySlot with stale frozen slot outputs (wrong dtypes).
  // phaseWarmup resets all slot states and re-derives shapes from live inputs.
  plan->resetExecuteCount();
  // Reset frozen constant detection so detectFrozenConstants() re-runs
  // after the post-invalidation warmup.  Without this, stale frozen
  // classifications persist — slots marked FROZEN_CONSTANT keep their old
  // output arrays (potentially wrong dtype) and are skipped during execution.
  plan->resetFrozenConstantDetection();
}

}  // namespace SegmentLifecycle
}  // namespace graph
}  // namespace sd

// Cleanup macros — they use common identifiers that could collide
#undef SLS_ASSERT_FROM
#undef SLS_ASSERT_NOT_TERMINAL
