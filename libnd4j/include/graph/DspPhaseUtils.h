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
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Phase transition and lifecycle utilities for DSP execution.
// Centralizes plan phase naming, advancement helpers, and demotion logic.
//
// Segment phase predicates (segmentBlocksPlanPhase, segmentHasStablePointers,
// segmentIsFullyReplaying) are defined in NativeDynamicShapePlan.cpp because
// they require the full GraphSegment and NativeSlot definitions.
//

#ifndef LIBND4J_DSP_PHASE_UTILS_H
#define LIBND4J_DSP_PHASE_UTILS_H

#include <system/common.h>
#include <graph/DspDiagnostics.h>

#include <cstdint>
#include <cassert>

namespace sd {
namespace graph {

// ─── Forward declarations ────────────────────────────────────────────────
enum class PlanPhase : uint8_t;
// ExecutionPhase REMOVED — unified into SegmentLifecycleState

namespace dsp {

// ─── PlanPhase string helpers ─────────────────────────────────────────────

SD_INLINE const char* planPhaseName(PlanPhase phase) {
  static const char* names[] = {"SLOT_BY_SLOT", "SHAPES_FROZEN", "REPLAYING"};
  int idx = static_cast<int>(phase);
  return (idx >= 0 && idx < 3) ? names[idx] : "UNKNOWN";
}

// executionPhaseName REMOVED — use GraphSegmentExec::displayPhaseName() or
// SegmentLifecycle::stateName() instead

// ─── Status enum string helper (shared across translation units) ─────────
// Replaces duplicate statusName_gpu / statusName_seg switch tables.
SD_INLINE const char* dspStatusName(Status status) {
  return sd::statusName(status);
}

}  // namespace dsp
}  // namespace graph
}  // namespace sd

// ─── Phase guard macros ─────────────────────────────────────────────────────
//
// Enforce plan phase invariants at method entry points.
// Fires DSP_DIAG (FALLBACK category) + sd_printf + assert.
// These compile to nothing under __CUDA_ARCH__ (device code).
//
// Usage: Place as the FIRST line of a method body.
//   DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "buildSegments");
//   DSP_REQUIRE_PLAN_PHASE_EXACT(PlanPhase::SLOT_BY_SLOT, "phaseFreeze");
//   DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SHAPES_FROZEN, "someMethod");
//
// getPlanPhase() must be in scope (member of NativeDynamicShapePlan).

#ifndef __CUDA_ARCH__

#define DSP_REQUIRE_PLAN_PHASE_AT_MOST(maxPhase, methodName)                   \
  do {                                                                          \
    const auto _curPhase = getPlanPhase();                                      \
    if (_curPhase > (maxPhase)) {                                              \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires <= %s", \
               (methodName), dsp::planPhaseName(_curPhase),                    \
               dsp::planPhaseName(maxPhase));                                  \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires <= %d\n", \
                (methodName), (int)_curPhase, (int)(maxPhase));                \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

#define DSP_REQUIRE_PLAN_PHASE_EXACT(requiredPhase, methodName)                \
  do {                                                                          \
    const auto _curPhase = getPlanPhase();                                      \
    if (_curPhase != (requiredPhase)) {                                        \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires %s", \
               (methodName), dsp::planPhaseName(_curPhase),                    \
               dsp::planPhaseName(requiredPhase));                             \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires %d\n",  \
                (methodName), (int)_curPhase, (int)(requiredPhase));            \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

#define DSP_REQUIRE_PLAN_PHASE_AT_LEAST(minPhase, methodName)                  \
  do {                                                                          \
    const auto _curPhase = getPlanPhase();                                      \
    if (_curPhase < (minPhase)) {                                              \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires >= %s", \
               (methodName), dsp::planPhaseName(_curPhase),                    \
               dsp::planPhaseName(minPhase));                                  \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires >= %d\n", \
                (methodName), (int)_curPhase, (int)(minPhase));                \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

// ─── Segment lifecycle diagnostic event ──────────────────────────────────────
//
// DSP_SET_SEG_PHASE is now a diagnostic-only log. The actual lifecycle state
// is set by SegmentLifecycle:: transition functions in _gpubackend.cpp.
// This macro logs the current lifecycle state with a reason tag for grep/parse.
//
// Usage:
//   DSP_LOG_SEG_PHASE(segment, "cpu_graph_first_exec");

#define DSP_LOG_SEG_PHASE(SEG, REASON)                                         \
  do {                                                                          \
    DSP_DIAG(EXECUTE,                                                          \
             "[SEG_PHASE] seg[%d-%d] lifecycle=%s reason=%s execCount=%d",     \
             (SEG).def.startSlot, (SEG).def.endSlot,                           \
             (SEG).exec.displayPhaseName(), (REASON),                          \
             (SEG).exec.executionCount);                                        \
  } while (0)

// Legacy compatibility: DSP_SET_SEG_PHASE now just logs (no state assignment).
// Call sites that used to set ExecutionPhase should use SegmentLifecycle::
// transition functions for the actual state change, and this macro for logging.
#define DSP_SET_SEG_PHASE(SEG, NEW_PHASE_UNUSED, REASON)                       \
  DSP_LOG_SEG_PHASE(SEG, REASON)

#else  // __CUDA_ARCH__

#define DSP_REQUIRE_PLAN_PHASE_AT_MOST(maxPhase, methodName)  ((void)0)
#define DSP_REQUIRE_PLAN_PHASE_EXACT(requiredPhase, methodName) ((void)0)
#define DSP_REQUIRE_PLAN_PHASE_AT_LEAST(minPhase, methodName)  ((void)0)
#define DSP_LOG_SEG_PHASE(SEG, REASON) ((void)0)
#define DSP_SET_SEG_PHASE(SEG, NEW_PHASE_UNUSED, REASON) ((void)0)

#endif  // __CUDA_ARCH__

#endif  // LIBND4J_DSP_PHASE_UTILS_H
