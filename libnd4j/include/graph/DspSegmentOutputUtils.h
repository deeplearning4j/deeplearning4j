/* ****************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 * ****************************************************************************/

#ifndef LIBND4J_DSP_SEGMENT_OUTPUT_UTILS_H
#define LIBND4J_DSP_SEGMENT_OUTPUT_UTILS_H

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include <system/common.h>

namespace sd {
namespace graph {
namespace dsp {

/**
 * Visit actual input arrays an op declares writable. This is the same write
 * contract for direct compiled execution and replay; no op-name classification
 * or blanket variable-input dirty marking belongs in either path.
 * Callback receives (input ordinal, encoded source index, array).
 */
template <typename SlotT, typename ArrayT, typename Callback>
SD_INLINE void forEachOpWrittenInput(const SlotT& slot,
                                     ArrayT** externalInputs, int numExternalInputs,
                                     ArrayT** outputSlots, int totalOutputSlots,
                                     Callback&& callback) {
  if (slot.ident.op == nullptr || slot.wiring.inputSourceIndices == nullptr) return;
  const auto* descriptor = slot.ident.op->getOpDescriptor();
  if (descriptor == nullptr) return;
  auto resolve = [&](int input) -> ArrayT* {
    if (input < 0 || input >= slot.wiring.numInputs) return nullptr;
    const int source = slot.wiring.inputSourceIndices[input];
    if (source >= 0) {
      return outputSlots != nullptr && source < totalOutputSlots ? outputSlots[source] : nullptr;
    }
    const auto external = -static_cast<int64_t>(source) - 1;
    return externalInputs != nullptr && external < numExternalInputs ? externalInputs[external] : nullptr;
  };
  for (const auto& group : descriptor->getInputWriteGroups()) {
    bool active = true;
    for (const auto& required : group.requiredInputs) {
      auto* input = resolve(required.first);
      if (input == nullptr || input->isEmpty() || input->lengthOf() == 0 ||
          input->rankOf() < required.second) { active = false; break; }
    }
    if (!active) continue;
    if (group.typeInput >= 0) {
      auto* input = resolve(group.typeInput);
      if (input == nullptr || input->dataType() != group.requiredType) continue;
    }
    for (int written : group.indices) {
      auto* input = resolve(written);
      if (input != nullptr && !input->isEmpty()) {
        callback(written, slot.wiring.inputSourceIndices[written], input);
      }
    }
  }
}

/**
 * Visit the flat output-slot indices produced by an operation range.
 *
 * GraphSegmentDef::startSlot/endSlot are execution-operation indices.  They
 * must never be used as indices into NativeDynamicShapePlan::outputSlots_,
 * whose entries are flattened across all operation outputs.  Keeping this
 * conversion in one helper makes the two namespaces explicit at call sites.
 *
 * The callback receives (operation index, flat output-slot index, output ordinal).
 */
template <typename SlotT, typename Callback>
SD_INLINE void forEachSegmentOutputSlot(const SlotT* slots, int numSlots,
                                        int firstOp, int lastOp,
                                        int totalOutputSlots, Callback&& callback) {
  if (slots == nullptr || numSlots <= 0 || totalOutputSlots <= 0 || lastOp < firstOp) {
    return;
  }

  const int begin = std::max(0, firstOp);
  const int end = std::min(lastOp, numSlots - 1);
  for (int opIndex = begin; opIndex <= end; ++opIndex) {
    const auto& wiring = slots[opIndex].wiring;
    if (wiring.outputSlotIndices == nullptr || wiring.numOutputs <= 0) continue;
    for (int outputOrdinal = 0; outputOrdinal < wiring.numOutputs; ++outputOrdinal) {
      const int outputSlot = wiring.outputSlotIndices[outputOrdinal];
      if (outputSlot < 0 || outputSlot >= totalOutputSlots) continue;
      callback(opIndex, outputSlot, outputOrdinal);
    }
  }
}

/**
 * Resolve the first output of the last operation in an operation range.
 * Returns -1 when the range has no valid output.  There is deliberately no
 * fallback to the operation index: that would reintroduce the namespace bug.
 */
template <typename SlotT>
SD_INLINE int firstSegmentOutputSlot(const SlotT* slots, int numSlots,
                                     int firstOp, int lastOp,
                                     int totalOutputSlots) {
  if (slots == nullptr || numSlots <= 0 || totalOutputSlots <= 0 || lastOp < firstOp) {
    return -1;
  }

  const int begin = std::max(0, firstOp);
  const int end = std::min(lastOp, numSlots - 1);
  for (int opIndex = end; opIndex >= begin; --opIndex) {
    const auto& wiring = slots[opIndex].wiring;
    if (wiring.outputSlotIndices == nullptr || wiring.numOutputs <= 0) continue;
    for (int outputOrdinal = 0; outputOrdinal < wiring.numOutputs; ++outputOrdinal) {
      const int outputSlot = wiring.outputSlotIndices[outputOrdinal];
      if (outputSlot >= 0 && outputSlot < totalOutputSlots) return outputSlot;
    }
  }
  return -1;
}

/** Find the producing operation for one flattened output slot in a range. */
template <typename SlotT>
SD_INLINE int findSegmentOutputProducer(const SlotT* slots, int numSlots,
                                        int firstOp, int lastOp,
                                        int totalOutputSlots, int outputSlot) {
  int producer = -1;
  forEachSegmentOutputSlot(
      slots, numSlots, firstOp, lastOp, totalOutputSlots,
      [&](int opIndex, int candidate, int) {
        if (candidate == outputSlot && producer < 0) producer = opIndex;
      });
  return producer;
}

/**
 * Collect output values that a backend must inspect after functional warmup.
 *
 * A backend that requests precommit functional warmup calibrates from the
 * value produced by each admitted operation. A later in-place consumer must
 * not overwrite that output before backend compilation reads it. Capability
 * remains backend-owned; this helper only turns the declared planning policy
 * into a concrete output-slot lifetime set.
 */
template <typename SlotT, typename BackendT, typename RequestT>
SD_INLINE std::unordered_set<int> collectPrecommitCalibrationOutputSlots(
    const RequestT& request, const std::vector<BackendT*>& candidates,
    SlotT* slots, int numSlots, int totalOutputSlots) {
  std::unordered_set<int> protectedOutputs;
  if (slots == nullptr || numSlots <= 0 || totalOutputSlots <= 0) {
    return protectedOutputs;
  }

  for (BackendT* backend : candidates) {
    if (backend == nullptr ||
        !backend->planningPolicy(request).requiresPrecommitFunctionalWarmup) {
      continue;
    }
    for (int opIndex = 0; opIndex < numSlots; ++opIndex) {
      if (!backend->canResolveSlot(request, slots, opIndex)) continue;
      const auto& wiring = slots[opIndex].wiring;
      if (wiring.outputSlotIndices == nullptr || wiring.numOutputs <= 0) continue;
      for (int outputOrdinal = 0; outputOrdinal < wiring.numOutputs;
           ++outputOrdinal) {
        const int outputSlot = wiring.outputSlotIndices[outputOrdinal];
        if (outputSlot >= 0 && outputSlot < totalOutputSlots) {
          protectedOutputs.insert(outputSlot);
        }
      }
    }
  }
  return protectedOutputs;
}

/** Disable consumers that would overwrite a precommit calibration value. */
template <typename SlotT>
SD_INLINE int disableInPlaceConsumersOfSlots(
    SlotT* slots, int numSlots,
    const std::unordered_set<int>& protectedOutputs) {
  if (slots == nullptr || numSlots <= 0 || protectedOutputs.empty()) return 0;
  int disabled = 0;
  for (int opIndex = 0; opIndex < numSlots; ++opIndex) {
    auto& slot = slots[opIndex];
    if (protectedOutputs.count(slot.inPlaceSourceSlot()) == 0) continue;
    slot.disableInPlaceFusion();
    ++disabled;
  }
  return disabled;
}

}  // namespace dsp
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_SEGMENT_OUTPUT_UTILS_H
