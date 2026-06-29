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

// Initialize or reset segment to BUILDING:WARMUP (no prior-state assertion).
// Used at segment construction time, phaseWarmup init, and slot-by-slot fallback
// where the segment may be in any state. This is the ONLY entry point for
// raw segPhase.reset() + lifecycleState = NEEDS_WARMUP outside of
// invalidateSegmentCaptures / evictSegmentCapture (which do additional cleanup).
static inline void initSegmentPhase(GraphSegmentExec& exec,
                                    int startSlot = -1, int endSlot = -1) {
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (init seg[%d-%d])",
           exec.segPhase.displayName(), startSlot, endSlot);
  exec.segPhase.reset();  // PRIMARY: BUILDING:WARMUP
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
}

// BUILDING:WARMUP -> BUILDING:CAPTURING (skip compile for emulated/CUDA_GRAPHS modes)
// Used by EMULATED_REPLAY and CUDA_GRAPHS which don't have a separate compilation step.
static inline void skipToCapturing(GraphSegmentExec& exec, const char* backendName,
                                   int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::NEEDS_WARMUP, "skipToCapturing");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:CAPTURING (skip_compile backend=%s "
           "seg[%d-%d] execCount=%d)",
           exec.segPhase.displayName(), backendName, startSlot, endSlot,
           exec.executionCount);
  exec.segPhase.skipCompileToCapturing();  // PRIMARY: WARMUP → CAPTURING
  exec.compiledByBackend = backendName;
  exec.lifecycleState = SLS::CAPTURE_PENDING;  // Legacy sync
}

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
// gapsCaptured: true when native-only capture included gap ops (cuBLAS etc.)
// in the monolithic graph. The frozen fast path checks hasGapsInGraph() to
// know monolithic replay covers ALL ops — no live gap execution needed.
static inline void markCaptured(GraphSegmentExec& exec,
                                LongType inputAddrKey, LongType createValueKey,
                                LongType slotAddrHash, const char* backendName,
                                bool gapsCaptured = false) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markCaptured");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> SEALED (backend=%s inputAddrKey=%lld "
           "createValueKey=%lld slotAddrHash=%lld execCount=%d needsArgRefresh=%d "
           "gapsCaptured=%d)",
           exec.segPhase.displayName(), backendName,
           (long long)inputAddrKey, (long long)createValueKey,
           (long long)slotAddrHash, exec.executionCount, (int)exec.needsArgRefresh(),
           (int)gapsCaptured);
  exec.segPhase.seal();  // PRIMARY: BUILDING:CAPTURING → SEALED
  exec.outcome = SegmentExecOutcome::GRAPH_REPLAY;
  exec.sealCapture(inputAddrKey, createValueKey, slotAddrHash, backendName, gapsCaptured);
  exec.lifecycleState = SLS::REPLAYING;  // Legacy sync
  exec.handleTracker.record(ReplayHandleEvent::Kind::INSTANTIATE, exec.executionCount,
                            0, 0, backendName);
}

// Legacy compatibility — markReplaying is now a no-op since markCaptured
// transitions directly to SEALED. Callers that previously did
// CAPTURED→REPLAYING on first replay are harmless no-ops.
static inline void markReplaying(GraphSegmentExec& exec) {
  (void)exec;
}

// any -> FAILED (terminal)
static inline void markFailed(GraphSegmentExec& exec, const char* reason,
                              int startSlot = -1, int endSlot = -1) {
  sd_printf("DSP ERROR: seg[%d-%d] capture/compilation FAILED. "
            "reason=%s — permanent slot-by-slot fallback.\n",
            startSlot, endSlot, reason ? reason : "?");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> FAILED (reason=%s execCount=%d compiledBy=%s)",
           exec.segPhase.displayName(), reason, exec.executionCount,
           exec.compiledByBackend.c_str());
  const char* prevPhase = exec.segPhase.displayName();
  exec.segPhase.fail();  // PRIMARY
  exec.compilationFailed = true;
  exec.outcome = SegmentExecOutcome::COMPILE_FAILED;
  exec.terminalReason = reason;
  exec.lifecycleState = SLS::FAILED;  // Legacy sync
  exec.handleTracker.record(ReplayHandleEvent::Kind::EXEC_ERROR, exec.executionCount,
                            0, 0, reason);
  DspDiagnostics::getInstance().recordSegmentTerminal(
      startSlot, endSlot, exec.executionCount,
      static_cast<int>(exec.outcome), prevPhase, reason,
      exec.compiledByBackend.c_str());
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
  exec.handleTracker.record(ReplayHandleEvent::Kind::OOM_DEFERRED, exec.executionCount,
                            0, 0, "instantiate_oom");
}

// ── Terminal outcomes via self-contained lifecycle methods ────────────────
// These replace all raw seg.exec.outcome / seg.exec.captureProducedNoKernels
// assignments in dispatch code. Every terminal state must go through one of these.

// BUILDING:CAPTURING -> SEALED (zero-kernel: capture genuinely produced 0 GPU nodes)
// Terminal: segment executes slot-by-slot forever. Only valid when capture completed
// but the graph contained 0 nodes (view/identity/shape ops only).
static inline void markZeroKernel(GraphSegmentExec& exec, const char* reason,
                                  int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markZeroKernel");
  sd_printf("DSP WARN: seg[%d-%d] CUDA graph captured ZERO nodes. "
            "This segment will ALWAYS run slot-by-slot. reason=%s\n",
            startSlot, endSlot, reason ? reason : "?");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> SEALED:ZERO_KERNEL (reason=%s execCount=%d)",
           exec.segPhase.displayName(), reason, exec.executionCount);
  const char* prevPhase = exec.segPhase.displayName();
  exec.segPhase.seal();  // PRIMARY: BUILDING:CAPTURING → SEALED
  exec.captureProducedNoKernels = true;
  exec.outcome = SegmentExecOutcome::ZERO_KERNEL_SBS;
  exec.terminalReason = reason;
  exec.lifecycleState = SLS::REPLAYING;  // Legacy sync
  exec.handleTracker.record(ReplayHandleEvent::Kind::DESTROY, exec.executionCount,
                            0, 0, reason);
  DspDiagnostics::getInstance().recordSegmentTerminal(
      startSlot, endSlot, exec.executionCount,
      static_cast<int>(exec.outcome), prevPhase, reason,
      exec.compiledByBackend.c_str());
}

// any BUILDING -> SEALED (not fusible: no backend can fuse this segment's ops)
// Terminal: segment executes slot-by-slot forever. Expected for permute/reshape/identity
// segments where compilation cascade found zero fusible backends.
static inline void markNotFusible(GraphSegmentExec& exec, const char* reason,
                                  int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_NOT_TERMINAL(exec, "markNotFusible");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> SEALED:NOT_FUSIBLE (reason=%s execCount=%d)",
           exec.segPhase.displayName(), reason, exec.executionCount);
  const char* prevPhase = exec.segPhase.displayName();
  exec.segPhase.sealNonCapture();  // PRIMARY: any BUILDING → SEALED
  exec.noFusibleOps = true;
  exec.outcome = SegmentExecOutcome::NOT_FUSIBLE;
  exec.terminalReason = reason;
  exec.lifecycleState = SLS::REPLAYING;  // Legacy sync
  DspDiagnostics::getInstance().recordSegmentTerminal(
      startSlot, endSlot, exec.executionCount,
      static_cast<int>(exec.outcome), prevPhase, reason,
      exec.compiledByBackend.c_str());
}

// BUILDING:CAPTURING -> SEALED (emulated replay sealed to steady state)
// For EMULATED_REPLAY backend: slot-by-slot execution by design, not by failure.
static inline void markEmulatedSealed(GraphSegmentExec& exec,
                                      int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markEmulatedSealed");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> SEALED:EMULATED (execCount=%d)",
           exec.segPhase.displayName(), exec.executionCount);
  const char* prevPhase = exec.segPhase.displayName();
  exec.segPhase.seal();  // PRIMARY: BUILDING:CAPTURING → SEALED
  exec.outcome = SegmentExecOutcome::ZERO_KERNEL_SBS;
  exec.terminalReason = "emulated_replay_sealed";
  exec.lifecycleState = SLS::REPLAYING;  // Legacy sync
  DspDiagnostics::getInstance().recordSegmentTerminal(
      startSlot, endSlot, exec.executionCount,
      static_cast<int>(exec.outcome), prevPhase, "emulated_replay_sealed",
      exec.compiledByBackend.c_str());
}

// OOM_DEFERRED -> CAPTURE_PENDING (OOM retry fires — back to capture attempt)
// Called when executionCount reaches the retry threshold. Clears OOM flag only,
// preserves compilation state so capture can proceed.
static inline void markOomRetryFiring(GraphSegmentExec& exec,
                                      int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::OOM_DEFERRED, "markOomRetryFiring");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:CAPTURING (oom_retry_firing execCount=%d "
           "retryAfter=%d retries=%d)",
           exec.segPhase.displayName(), exec.executionCount,
           exec.segPhase.oomRetryAfterExec, exec.segPhase.oomRetryCount);
  exec.segPhase.clearOomRetry();  // PRIMARY: OOM_RETRY → CAPTURING
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.lifecycleState = SLS::CAPTURE_PENDING;  // Legacy sync
}

// CAPTURE_PENDING -> CAPTURE_PENDING (capture bail-out due to thread contention)
// Called when another thread holds the capture lock. Resets replay handle and
// decrements executionCount so next call re-attempts capture.
static inline void markCaptureBailout(GraphSegmentExec& exec,
                                      int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markCaptureBailout");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> CAPTURE_PENDING (capture_bailout execCount=%d)",
           exec.segPhase.displayName(), exec.executionCount);
  exec.handleTracker.record(ReplayHandleEvent::Kind::INVALIDATE, exec.executionCount,
                            0, 0, "capture_bailout");
  exec.replayHandle.reset();
  exec.outcome = SegmentExecOutcome::PENDING;
  // Decrement so next call re-attempts capture
  if (exec.executionCount > 0) exec.executionCount--;
  // Phase stays CAPTURE_PENDING — no demotion
  exec.lifecycleState = SLS::CAPTURE_PENDING;  // Legacy sync
}

// SEALED/REPLAYING -> NEEDS_WARMUP (evict a segment's graph to free GPU memory)
// Called during OOM recovery to evict a smaller segment's graph so a larger one
// can be instantiated. Resets the evicted segment to re-capture on future execution.
static inline void evictSegmentCapture(GraphSegmentExec& exec,
                                       int startSlot = -1, int endSlot = -1) {
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (evict_for_oom seg[%d-%d] execCount=%d)",
           exec.segPhase.displayName(), startSlot, endSlot, exec.executionCount);
  // Release GPU resources from the replay handle before resetting
  if (exec.replayHandle) {
    exec.handleTracker.record(ReplayHandleEvent::Kind::DESTROY, exec.executionCount,
                              0, 0, "evict_for_oom");
    exec.replayHandle->releaseWorkspace(nullptr, startSlot);
    exec.replayHandle->freeHostPointers();
    exec.replayHandle->clearExternalAddresses();
    exec.replayHandle.reset();
  }
  exec.handleTracker.record(ReplayHandleEvent::Kind::INVALIDATE, exec.executionCount,
                            0, 0, "evict_for_oom");
  exec.segPhase.reset();  // PRIMARY: back to BUILDING:WARMUP
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.resetCaptureKeys();
  exec.compilationFailed = false;
  exec.gapOpsCapturedInGraph = false;
  exec.markArgsStale();
  exec.compiledByBackend.clear();
  exec.executionCount = 0;
  exec.lastReplayExecCount = 0;
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
}

// CAPTURE_PENDING -> NEEDS_WARMUP (functional capture failed — CPU stubs path)
// Called when CPU-stub functional capture endCapture/finalize fails.
// Resets to warmup for retry without marking terminal failure.
static inline void markFunctionalCaptureFailure(GraphSegmentExec& exec,
                                                 int startSlot = -1, int endSlot = -1) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markFunctionalCaptureFailure");
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (functional_capture_failed execCount=%d)",
           exec.segPhase.displayName(), exec.executionCount);
  exec.handleTracker.record(ReplayHandleEvent::Kind::EXEC_ERROR, exec.executionCount,
                            0, 0, "functional_capture_failed");
  exec.replayHandle.reset();
  exec.segPhase.reset();  // PRIMARY: back to BUILDING:WARMUP
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.markArgsStale();
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
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
  exec.handleTracker.record(ReplayHandleEvent::Kind::INVALIDATE, exec.executionCount,
                            0, 0, reason);
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
  exec.terminalReason = nullptr;
  exec.resetCaptureKeys();
  exec.compilationFailed = false;
  exec.markArgsStale();
  exec.compiledByBackend.clear();
  exec.executionCount = 0;
  exec.lastReplayExecCount = 0;
  exec.lifecycleState = SLS::NEEDS_WARMUP;  // Legacy sync
  // NOTE: intentionally do NOT call plan->resetExecuteCount() here.
  // The plan shapes are unchanged — only segment CUDA graph captures are stale.
  // Resetting executeCount would cause isFirstFrozenWarmup=true → phaseWarmup
  // fires destructively, destroying all CUDA graphs and producing zeros.
}

// Recoverable capture error — reset to BUILDING:WARMUP for retry.
// Called when CUDA capture fails (e.g. endCapture error) but the CUDA context
// is still valid. Uses invalidateSegmentCaptures for a clean reset.
// Does NOT set captureProducedNoKernels — this was an error, not a zero-kernel segment.
static inline void markCaptureErrorRetry(NativeDynamicShapePlan* plan, GraphSegment& seg,
                                          const char* reason) {
  DSP_DIAG(EXECUTE, "LIFECYCLE: %s -> BUILDING:WARMUP (capture_error_retry: %s execCount=%d)",
           seg.exec.segPhase.displayName(), reason, seg.exec.executionCount);
  invalidateSegmentCaptures(plan, seg, reason);
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
  exec.handleTracker.record(ReplayHandleEvent::Kind::INVALIDATE, exec.executionCount,
                            0, 0, reason);
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
  exec.terminalReason = nullptr;
  exec.resetCaptureKeys();
  exec.compilationFailed = false;
  exec.markArgsStale();
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

// ── Targeted field operations ────────────────────────────────────────────
// These exist so that NO call site outside this header writes compilationFailed,
// outcome, or captureProducedNoKernels directly. Each method documents WHY
// the mutation is allowed.

// Clear compilationFailed after a successful capture recovers from a prior
// Triton compilation failure. This is NOT a full lifecycle transition — the
// segment continues through the normal markCaptured path. We only clear the
// flag so cleanup doesn't treat the segment as non-graph-managed.
static inline void clearCompilationFailedOnRecovery(GraphSegmentExec& exec,
                                                     const char* reason,
                                                     int startSlot = -1, int endSlot = -1) {
  if (!exec.compilationFailed) return;  // no-op if not failed
  DSP_DIAG(COMPILE, "LIFECYCLE: clearing compilationFailed for seg[%d-%d] reason=%s",
           startSlot, endSlot, reason);
  exec.compilationFailed = false;
}

// Copy compilation state from one segment exec to another. Used when dispatch
// creates ephemeral gapSeg or warmupSeg copies that need to inherit the
// compilation status of the parent segment.
static inline void copyCompilationState(GraphSegmentExec& dst, const GraphSegmentExec& src) {
  dst.compilationFailed = src.compilationFailed;
}

// Reset outcome and compilationFailed for GPU resource teardown functions
// (platformCleanupSegmentForRebuild, platformFreePlanResources,
//  platformReleaseSegmentGpuResources). These are called when GPU resources
// are being freed and the segment returns to a clean state.
static inline void resetForResourceRelease(GraphSegmentExec& exec) {
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.compilationFailed = false;
}

// Reset outcome and compilationFailed for cache invalidation
// (invalidatePlanSegmentCache, invalidatePlanBackendCaches in NativeOps_dsp).
// Called from the JNI layer when segment backend caches need to be cleared.
static inline void resetForCacheInvalidation(GraphSegmentExec& exec) {
  exec.outcome = SegmentExecOutcome::PENDING;
  exec.compilationFailed = false;
}

}  // namespace SegmentLifecycle

// ── Replay-verify diagnostic guard ────────────────────────────────────────
// RAII guard that temporarily overrides compilationFailed and executionCount
// to force slot-by-slot re-execution for replay verification. Restores original
// state on destruction. This centralizes the save/restore pattern so no call
// site directly writes compilationFailed outside of lifecycle methods.
struct ReplayVerifyStateGuard {
  GraphSegmentExec& exec;
  int savedExecCount;
  bool savedCompilationFailed;

  ReplayVerifyStateGuard(GraphSegmentExec& e) : exec(e) {
    savedExecCount = exec.executionCount;
    savedCompilationFailed = exec.compilationFailed;
    exec.compilationFailed = true;    // Force slot-by-slot
    exec.executionCount = 999;        // Skip warmup thresholds
  }
  ~ReplayVerifyStateGuard() {
    exec.executionCount = savedExecCount;
    exec.compilationFailed = savedCompilationFailed;
  }
  // Non-copyable
  ReplayVerifyStateGuard(const ReplayVerifyStateGuard&) = delete;
  ReplayVerifyStateGuard& operator=(const ReplayVerifyStateGuard&) = delete;
};

}  // namespace graph
}  // namespace sd

// Cleanup macros — they use common identifiers that could collide
#undef SLS_ASSERT_FROM
#undef SLS_ASSERT_NOT_TERMINAL
