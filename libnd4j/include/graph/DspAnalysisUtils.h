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
// Centralized analysis utilities for DSP infrastructure.
// Replaces ad-hoc cross-section intermediate detection and buffer alias
// detection that was duplicated across TritonIRBuilder_module.cpp.
//

#ifndef LIBND4J_DSP_ANALYSIS_UTILS_H
#define LIBND4J_DSP_ANALYSIS_UTILS_H

#include <ops/declarable/OpDescriptor.h>
#include <graph/NativeDynamicShapePlan.h>
#include <system/common.h>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {
namespace dsp {

// ─── Cross-section intermediate detection ────────────────────────────────────
//
// A cross-section intermediate is a slot output that is:
//   1. Produced in one KernelSection
//   2. Consumed in a DIFFERENT KernelSection within the same kernel
//
// These require global-memory buffers because values cannot be forwarded
// through SSA registers across section boundaries.

template <typename SectionT>
SD_INLINE std::unordered_set<int> identifyCrossSectionIntermediates(
    NativeSlot* slots, int startSlot, int endSlot,
    const std::vector<SectionT>& sections) {

  // Build set of all slot outputs produced within the kernel range
  std::unordered_set<int> internalOutputs;
  for (int s = startSlot; s <= endSlot; s++) {
    for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
      internalOutputs.insert(slots[s].wiring.outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> result;
  for (size_t si = 0; si < sections.size(); si++) {
    auto& sec = sections[si];
    for (int i = sec.startSlot; i <= sec.endSlot; i++) {
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int srcIdx = slots[i].wiring.inputSourceIndices[inp];
        if (srcIdx < 0) continue;  // External input

        // Check if produced in a DIFFERENT section
        bool producedInThisSection = false;
        for (int j = sec.startSlot; j <= sec.endSlot; j++) {
          for (int o = 0; o < slots[j].wiring.numOutputs; o++) {
            if (slots[j].wiring.outputSlotIndices[o] == srcIdx) {
              producedInThisSection = true;
              break;
            }
          }
          if (producedInThisSection) break;
        }
        if (!producedInThisSection && internalOutputs.count(srcIdx)) {
          result.insert(srcIdx);
        }
      }
    }
  }
  return result;
}

// ─── Producer section lookup ─────────────────────────────────────────────────
//
// Maps each internal slot output to the section index that produces it.

template <typename SectionT>
SD_INLINE std::unordered_map<int, int> buildProducerSectionMap(
    NativeSlot* slots,
    const std::vector<SectionT>& sections,
    const std::unordered_set<int>& internalOutputs) {

  std::unordered_map<int, int> result;
  for (size_t secIdx = 0; secIdx < sections.size(); secIdx++) {
    for (int si = sections[secIdx].startSlot; si <= sections[secIdx].endSlot; si++) {
      for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
        int outIdx = slots[si].wiring.outputSlotIndices[o];
        if (internalOutputs.count(outIdx)) {
          result[outIdx] = static_cast<int>(secIdx);
        }
      }
    }
  }
  return result;
}

// ─── Buffer alias detection ──────────────────────────────────────────────────
//
// Builds a set of GPU buffer addresses from input args. Used to detect when
// an output slot shares the same GPU buffer as an input (view aliasing).
// Storing to an aliased output creates a data race — different warps may
// execute the store before other warps read the aliased input.

template <typename ArgT>
SD_INLINE std::unordered_set<uintptr_t> buildInputBufferAddressSet(
    const std::vector<ArgT>& inputArgs,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalInputs, int numExternalInputs) {

  std::unordered_set<uintptr_t> result;
  for (auto& inArg : inputArgs) {
    NDArray* inArr = nullptr;
    if (inArg.slotIndex < 0) {
      int ei = -(inArg.slotIndex + 1);
      if (ei < numExternalInputs && externalInputs[ei]) inArr = externalInputs[ei];
    } else {
      if (inArg.slotIndex < totalOutputSlots && outputSlots && outputSlots[inArg.slotIndex])
        inArr = outputSlots[inArg.slotIndex];
    }
    if (inArr && inArr->specialBuffer()) {
      result.insert(reinterpret_cast<uintptr_t>(inArr->specialBuffer()));
    }
  }
  return result;
}

// Check if a specific output slot's buffer aliases any input buffer.
SD_INLINE bool isOutputAliasedWithInput(
    int outSlotIdx,
    NDArray** outputSlots, int totalOutputSlots,
    const std::unordered_set<uintptr_t>& inputBufferAddrs) {

  if (!outputSlots || outSlotIdx < 0 || outSlotIdx >= totalOutputSlots) return false;
  NDArray* outArr = outputSlots[outSlotIdx];
  if (!outArr || !outArr->specialBuffer()) return false;
  uintptr_t outAddr = reinterpret_cast<uintptr_t>(outArr->specialBuffer());
  return inputBufferAddrs.count(outAddr) > 0;
}

// ─── Slot trait resolution ───────────────────────────────────────────────────
//
// Resolves the combined op traits for a slot from both the op descriptor
// and the slot's cached flags. Centralizes the manual bit-setting pattern
// that was duplicated in TritonGraphBackend_compile.cu.

SD_INLINE uint32_t resolveSlotTraits(const NativeSlot& slot) {
  return slot.opTraits();
}

// ─── Slot/segment producer lookup ───────────────────────────────────────────
//
// Find which step in a segment range produces a given output slot index.
// Returns -1 if no producer found within the segment.

SD_INLINE int findProducerStepInSegment(const GraphSegment& seg, const NativeSlot* slots, int outputSlotIdx) {
  if (slots == nullptr || outputSlotIdx < 0) return -1;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    const auto& slot = slots[s];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      if (slot.wiring.outputSlotIndices[o] == outputSlotIdx) {
        return s;
      }
    }
  }
  return -1;
}

// Find which step across ALL slots produces a given output slot index.
// Returns -1 if no producer found.

SD_INLINE int findProducingStepForOutputSlot(const NativeSlot* slots, int numSlots, int outputSlotIdx) {
  if (slots == nullptr || outputSlotIdx < 0) return -1;
  for (int s = 0; s < numSlots; s++) {
    const auto& slot = slots[s];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      if (slot.wiring.outputSlotIndices[o] == outputSlotIdx) {
        return s;
      }
    }
  }
  return -1;
}

// Check if a segment has internal inputs to value-dependent-shape ops.
// This determines whether replay invariant tracking is needed.

SD_INLINE bool segmentHasInternalValueShapeInputs(const GraphSegment& seg, const NativeSlot* slots) {
  if (slots == nullptr) return false;

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    const uint32_t traits = resolveSlotTraits(slots[s]);
    if ((traits & sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) == 0) continue;

    for (int i = 0; i < slots[s].wiring.numInputs; i++) {
      const int srcIdx = slots[s].wiring.inputSourceIndices[i];
      if (srcIdx < 0) continue;
      if (findProducerStepInSegment(seg, slots, srcIdx) >= 0) {
        return true;
      }
    }
  }

  return false;
}

// Resolve an input source index to the corresponding NDArray.
// Positive indices are output slot references; negative indices encode
// external inputs as -(extIdx+1).

SD_INLINE NDArray* resolveInputSourceArray(int srcIdx,
                                           NDArray** outputSlots, int totalOutputSlots,
                                           NDArray** externalArrays, int numExt) {
  if (srcIdx < 0) {
    const int extIdx = -(srcIdx + 1);
    return (extIdx >= 0 && extIdx < numExt) ? externalArrays[extIdx] : nullptr;
  }
  return (srcIdx >= 0 && srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
}

}  // namespace dsp
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_ANALYSIS_UTILS_H
