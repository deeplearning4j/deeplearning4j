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

//
// Shared types and utilities for all graph backends (CPU and GPU).
//
// Extracts common definitions that were duplicated across 6+ backend headers:
//   - SegmentCacheKey / SegmentCacheHash: cache key for compiled segments
//   - ArgMapping: maps kernel buffer arguments to slot/external indices
//   - buildArgMappings(): reconstructs the 3-phase arg ordering used by
//     MLIR-based backends (CpuIRBuilder, MlxIRBuilder, ArmHybrid)
//

#ifndef LIBND4J_GRAPH_BACKEND_COMMON_H
#define LIBND4J_GRAPH_BACKEND_COMMON_H

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <system/common.h>

#include <functional>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

// ─── Segment cache key (used by all 6 CPU backends + GPU JIT backends) ───────

struct SegmentCacheKey {
  int startSlot;
  int endSlot;
  LongType shapeKey;
  bool operator==(const SegmentCacheKey& o) const {
    return startSlot == o.startSlot && endSlot == o.endSlot && shapeKey == o.shapeKey;
  }
};

struct SegmentCacheHash {
  size_t operator()(const SegmentCacheKey& k) const {
    size_t h = std::hash<int>()(k.startSlot);
    h ^= std::hash<int>()(k.endSlot) << 1;
    h ^= std::hash<LongType>()(k.shapeKey) << 2;
    return h;
  }
};

// ─── Argument mapping (used by MLIR-based backends: MLIR CPU, ARM Hybrid, MLX) ──

struct ArgMapping {
  int sourceIndex;  // <0: external input (-(idx+1)), >=0: outputSlot index
  bool isOutput;
};

// ─── 3-phase argument reconstruction ─────────────────────────────────────────
//
// All MLIR-based backends build kernel arguments in the same order:
//   Phase 1: External inputs and pre-segment sources
//   Phase 2: Externally visible outputs
//   Phase 3: Internal intermediate outputs
//
// The ComputeExternalOutputsFn callback abstracts over which IR builder's
// computeExternallyVisibleOutputs() to call (CpuIRBuilder vs MlxIRBuilder).

using ComputeExternalOutputsFn = std::function<std::unordered_set<int>(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots)>;

/**
 * Build the argument mapping vector for a compiled MLIR segment.
 *
 * @param slots           The slot array
 * @param startSlot       First slot in segment (inclusive)
 * @param endSlot         Last slot in segment (inclusive)
 * @param externalInputs  External input arrays
 * @param numExternalInputs Number of external inputs
 * @param outputSlots     Output slot arrays
 * @param totalOutputSlots Total output slots
 * @param totalSlots      Total slots in plan
 * @param computeExternalOutputs  Callback to compute externally visible outputs
 * @return Ordered vector of ArgMapping entries matching kernel arg order
 */
inline std::vector<ArgMapping> buildArgMappings(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int totalSlots,
    const ComputeExternalOutputsFn& computeExternalOutputs) {

  std::vector<ArgMapping> mappings;

  DSP_DIAG(BACKEND, "buildArgMappings: BEGIN [%d-%d] extInputs=%d outputSlots=%d totalSlots=%d",
           startSlot, endSlot, numExternalInputs, totalOutputSlots, totalSlots);

  // Collect all internal outputs (produced within this segment)
  std::unordered_set<int> internalOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenSrc;

  // Phase 1: Input args (external + pre-segment sources)
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (seenSrc.count(srcIdx)) continue;

      bool isExternal = (srcIdx < 0);
      bool isPreSegment = (!isExternal && !internalOutputs.count(srcIdx));

      if (isExternal || isPreSegment) {
        NDArray* arr = nullptr;
        if (isExternal) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
        } else {
          if (srcIdx < totalOutputSlots && outputSlots) arr = outputSlots[srcIdx];
        }
        if (!arr) continue;

        seenSrc.insert(srcIdx);
        mappings.push_back({srcIdx, false});
      }
    }
  }

  // Phase 2: Output args (externally visible)
  auto externalOutputSet = computeExternalOutputs(slots, startSlot, endSlot, totalSlots);
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      int outIdx = slots[i].wiring.outputSlotIndices[o];
      if (!externalOutputSet.count(outIdx)) continue;
      if (seenSrc.count(outIdx)) continue;
      seenSrc.insert(outIdx);

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      mappings.push_back({outIdx, true});
    }
  }

  // Phase 3: Internal intermediate outputs
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      int outIdx = slots[i].wiring.outputSlotIndices[o];
      if (seenSrc.count(outIdx)) continue;

      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) arr = outputSlots[outIdx];
      if (!arr) continue;

      seenSrc.insert(outIdx);
      mappings.push_back({outIdx, true});
    }
  }

  int inputCount = 0, outputCount = 0;
  for (const auto& m : mappings) {
    if (m.isOutput) outputCount++; else inputCount++;
  }
  DSP_DIAG(BACKEND, "buildArgMappings: END [%d-%d] total=%d (inputs=%d outputs=%d)",
           startSlot, endSlot, (int)mappings.size(), inputCount, outputCount);

  return mappings;
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_BACKEND_COMMON_H
