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
enum class ExecutionPhase : uint8_t;

namespace dsp {

// ─── PlanPhase string helpers ─────────────────────────────────────────────

SD_INLINE const char* planPhaseName(PlanPhase phase) {
  static const char* names[] = {"SLOT_BY_SLOT", "SHAPES_FROZEN",
                                "POINTERS_STABLE", "REPLAYING"};
  int idx = static_cast<int>(phase);
  return (idx >= 0 && idx < 4) ? names[idx] : "UNKNOWN";
}

SD_INLINE const char* executionPhaseName(ExecutionPhase phase) {
  static const char* names[] = {"WARMUP", "COMPILING", "COMPILED",
                                "REPLAYING", "SLOT_BY_SLOT"};
  int idx = static_cast<int>(phase);
  return (idx >= 0 && idx < 5) ? names[idx] : "UNKNOWN";
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
// planPhase_ must be in scope (member of NativeDynamicShapePlan).

#ifndef __CUDA_ARCH__

#define DSP_REQUIRE_PLAN_PHASE_AT_MOST(maxPhase, methodName)                   \
  do {                                                                          \
    if (planPhase_ > (maxPhase)) {                                             \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires <= %s", \
               (methodName), dsp::planPhaseName(planPhase_),                   \
               dsp::planPhaseName(maxPhase));                                  \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires <= %d\n", \
                (methodName), (int)planPhase_, (int)(maxPhase));               \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

#define DSP_REQUIRE_PLAN_PHASE_EXACT(requiredPhase, methodName)                \
  do {                                                                          \
    if (planPhase_ != (requiredPhase)) {                                       \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires %s", \
               (methodName), dsp::planPhaseName(planPhase_),                   \
               dsp::planPhaseName(requiredPhase));                             \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires %d\n",  \
                (methodName), (int)planPhase_, (int)(requiredPhase));           \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

#define DSP_REQUIRE_PLAN_PHASE_AT_LEAST(minPhase, methodName)                  \
  do {                                                                          \
    if (planPhase_ < (minPhase)) {                                             \
      DSP_DIAG(FALLBACK, "PHASE_VIOLATION: %s called in phase %s, requires >= %s", \
               (methodName), dsp::planPhaseName(planPhase_),                   \
               dsp::planPhaseName(minPhase));                                  \
      sd_printf("DSP PHASE VIOLATION: %s called in phase %d, requires >= %d\n", \
                (methodName), (int)planPhase_, (int)(minPhase));               \
      assert(false && "DSP phase violation");                                   \
    }                                                                           \
  } while (0)

#else  // __CUDA_ARCH__

#define DSP_REQUIRE_PLAN_PHASE_AT_MOST(maxPhase, methodName)  ((void)0)
#define DSP_REQUIRE_PLAN_PHASE_EXACT(requiredPhase, methodName) ((void)0)
#define DSP_REQUIRE_PLAN_PHASE_AT_LEAST(minPhase, methodName)  ((void)0)

#endif  // __CUDA_ARCH__

#endif  // LIBND4J_DSP_PHASE_UTILS_H
