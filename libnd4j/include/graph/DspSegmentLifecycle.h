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
    case SLS::CAPTURED:        return "CAPTURED";
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
//   CAPTURE_PENDING → CAPTURED          (markCaptured)
//   CAPTURED        → REPLAYING         (markReplaying)  — one capture, many replays
//   CAPTURE_PENDING → OOM_DEFERRED      (markOomDeferred)
//   any             → FAILED            (markFailed)     — terminal
//   any             → NEEDS_WARMUP      (invalidateForRebuild) — full reset
//
// The FAILED state is terminal: only invalidateForRebuild can leave it.
// REPLAYING is steady-state: only invalidateForRebuild can leave it.

#ifndef __CUDA_ARCH__
#define SLS_ASSERT_FROM(exec, expected, targetName) \
  do { \
    if (exec.lifecycleState != (expected)) { \
      sd_printf("DSP LIFECYCLE VIOLATION: %s requires state %s, but segment is in %s\n", \
                (targetName), stateName(expected), stateName(exec.lifecycleState)); \
      DSP_DIAG(FALLBACK, "LIFECYCLE_VIOLATION: %s requires %s, actual %s", \
               (targetName), stateName(expected), stateName(exec.lifecycleState)); \
      assert(false && "DSP segment lifecycle violation"); \
    } \
  } while (0)

#define SLS_ASSERT_NOT_TERMINAL(exec, targetName) \
  do { \
    if (exec.lifecycleState == SLS::FAILED) { \
      sd_printf("DSP LIFECYCLE VIOLATION: %s called on FAILED segment " \
                "(use invalidateForRebuild to reset)\n", (targetName)); \
      DSP_DIAG(FALLBACK, "LIFECYCLE_VIOLATION: %s on FAILED segment", (targetName)); \
      assert(false && "DSP transition from FAILED without invalidation"); \
    } \
  } while (0)
#else
#define SLS_ASSERT_FROM(exec, expected, targetName) ((void)0)
#define SLS_ASSERT_NOT_TERMINAL(exec, targetName) ((void)0)
#endif

// NEEDS_WARMUP -> NEEDS_COMPILE
static inline void markWarmupDone(GraphSegmentExec& exec) {
  SLS_ASSERT_FROM(exec, SLS::NEEDS_WARMUP, "markWarmupDone");
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "NEEDS_COMPILE", "");
  exec.lifecycleState = SLS::NEEDS_COMPILE;
}

// NEEDS_COMPILE -> CAPTURE_PENDING
static inline void markCompiled(GraphSegmentExec& exec, const char* backendName, LongType shapeKey) {
  SLS_ASSERT_FROM(exec, SLS::NEEDS_COMPILE, "markCompiled");
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "CAPTURE_PENDING",
                     "(backend=%s shapeKey=%lld)", backendName, (long long)shapeKey);
  exec.compiledByBackend = backendName;
  exec.lifecycleState = SLS::CAPTURE_PENDING;
}

// CAPTURE_PENDING -> CAPTURED
static inline void markCaptured(GraphSegmentExec& exec,
                                LongType inputAddrKey, LongType createValueKey,
                                LongType slotAddrHash, const char* backendName) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markCaptured");
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "CAPTURED",
                     "(backend=%s)", backendName);
  exec.capturedInputAddrKey = inputAddrKey;
  exec.capturedCreateValueKey = createValueKey;
  exec.capturedSlotAddrHash = slotAddrHash;
  exec.gapOpsCapturedInGraph = false;
  if (exec.compiledByBackend.empty()) exec.compiledByBackend = backendName;
  exec.lifecycleState = SLS::CAPTURED;
}

// CAPTURED -> REPLAYING
static inline void markReplaying(GraphSegmentExec& exec) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURED, "markReplaying");
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "REPLAYING", "");
  exec.lifecycleState = SLS::REPLAYING;
}

// any -> FAILED (terminal)
static inline void markFailed(GraphSegmentExec& exec, const char* reason) {
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "FAILED",
                     "(reason=%s)", reason);
  exec.compilationFailed = true;
  exec.lifecycleState = SLS::FAILED;
}

// CAPTURE_PENDING -> OOM_DEFERRED
static inline void markOomDeferred(GraphSegmentExec& exec, int retryAfterExec) {
  SLS_ASSERT_FROM(exec, SLS::CAPTURE_PENDING, "markOomDeferred");
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "OOM_DEFERRED",
                     "(retryAfter=%d retries=%d)", retryAfterExec, exec.captureOomRetries + 1);
  exec.captureOomRetries++;
  exec.captureRetryAfterExec = retryAfterExec;
  exec.lifecycleState = SLS::OOM_DEFERRED;
}

// any -> NEEDS_WARMUP (full invalidation)
static inline void invalidateForRebuild(NativeDynamicShapePlan* plan, GraphSegment& seg,
                                        const char* reason) {
  auto& exec = seg.exec;
  DSP_SEG_EVENT(seg, INVALIDATE, "reason=%s", reason);
  DSP_DIAG_STATE_TRANSITION(stateName, exec.lifecycleState, "NEEDS_WARMUP",
                     "(invalidate: %s)", reason);
  plan->cleanupSegmentForRebuild(seg, reason);
  exec.cachedShapeKey = 0;
  exec.capturedInputAddrKey = 0;
  exec.capturedCreateValueKey = 0;
  exec.capturedSlotAddrHash = 0;
  exec.compilationFailed = false;
  exec.argTableStable = false;
  exec.addrKeyStableCount = 0;
  exec.slotAddrStableCount = 0;
  exec.compiledByBackend.clear();
  exec.executionCount = 0;
  exec.lastReplayExecCount = 0;
  exec.lifecycleState = SLS::NEEDS_WARMUP;
}

}  // namespace SegmentLifecycle
}  // namespace graph
}  // namespace sd

// Cleanup macros — they use common identifiers that could collide
#undef SLS_ASSERT_FROM
#undef SLS_ASSERT_NOT_TERMINAL
