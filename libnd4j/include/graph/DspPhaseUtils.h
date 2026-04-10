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

#include <cstdint>

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

#endif  // LIBND4J_DSP_PHASE_UTILS_H
