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

#include <graph/DspLoopStructure.h>

#include <algorithm>
#include <map>
#include <set>

namespace sd {
namespace graph {

namespace {

struct LoopSlots {
  std::vector<int> merges;
  std::vector<int> switches;
  std::vector<int> nextIterations;
  std::vector<int> exits;
};

void setDead(const NativeSlot& slot, bool dead, bool* slotIsDead, int slotIsDeadSize) {
  for (int o = 0; o < slot.wiring.numOutputs; o++) {
    const int si = slot.wiring.outputSlotIndices[o];
    if (si >= 0 && si < slotIsDeadSize) slotIsDead[si] = dead;
  }
}

}  // namespace

bool dspBuildLoopRegions(NativeSlot* slots, int numSlots, int totalOutputSlots,
                         std::vector<LoopRegion>* regions, std::string* error) {
  regions->clear();
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    regions->clear();
    return false;
  };

  std::vector<int> producer(totalOutputSlots > 0 ? totalOutputSlots : 0, -1);
  for (int s = 0; s < numSlots; s++) {
    for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
      const int si = slots[s].wiring.outputSlotIndices[o];
      if (si >= 0 && si < totalOutputSlots) producer[si] = s;
    }
  }
  auto producerOf = [&](int source) { return source >= 0 && source < totalOutputSlots ? producer[source] : -1; };
  auto isType = [&](int step, ControlFlowType type) { return step >= 0 && slots[step].cf.controlFlowType == type; };
  auto isLoopMerge = [&](int step) {
    if (!isType(step, CF_MERGE)) return false;
    for (int i = 0; i < slots[step].wiring.numInputs; i++) {
      if (isType(producerOf(slots[step].wiring.inputSourceIndices[i]), CF_NEXT_ITERATION)) return true;
    }
    return false;
  };

  for (int s = 0; s < numSlots; s++) {
    slots[s].cf.loopBackTarget = -1;
    slots[s].cf.loopRegionIndex = -1;
  }

  // A loop's Merges are those its Switches read; its Switches share the loop predicate.
  std::map<int, LoopSlots> loopsByPredicate;
  std::map<int, int> predicateOfMerge;
  for (int s = 0; s < numSlots; s++) {
    if (!isType(s, CF_SWITCH) || slots[s].wiring.numInputs < 2 || slots[s].wiring.numOutputs < 2) continue;
    const int merge = producerOf(slots[s].wiring.inputSourceIndices[0]);
    if (!isLoopMerge(merge)) continue;
    const int predicate = slots[s].wiring.inputSourceIndices[1];
    auto known = predicateOfMerge.find(merge);
    if (known != predicateOfMerge.end() && known->second != predicate) {
      return fail("loop Merge '" + slots[merge].ident.opName + "' at slot " + std::to_string(merge) +
                  " feeds Switches of two different predicates");
    }
    auto& loop = loopsByPredicate[predicate];
    if (known == predicateOfMerge.end()) {
      predicateOfMerge[merge] = predicate;
      loop.merges.push_back(merge);
    }
    loop.switches.push_back(s);
  }
  for (int s = 0; s < numSlots; s++) {
    if (isLoopMerge(s) && predicateOfMerge.count(s) == 0) {
      return fail("loop Merge '" + slots[s].ident.opName + "' at slot " + std::to_string(s) +
                  " feeds no Switch");
    }
  }

  std::vector<LoopSlots> loops;
  for (auto& entry : loopsByPredicate) {
    LoopSlots& loop = entry.second;
    std::set<int> mergeInputs;
    for (int merge : loop.merges) {
      for (int i = 0; i < slots[merge].wiring.numInputs; i++) mergeInputs.insert(slots[merge].wiring.inputSourceIndices[i]);
    }
    std::set<int> falseOutputs;
    for (int sw : loop.switches) falseOutputs.insert(slots[sw].wiring.outputSlotIndices[0]);
    for (int s = 0; s < numSlots; s++) {
      if (isType(s, CF_NEXT_ITERATION) && slots[s].wiring.numOutputs > 0 &&
          mergeInputs.count(slots[s].wiring.outputSlotIndices[0]) != 0) {
        loop.nextIterations.push_back(s);
      } else if (isType(s, CF_EXIT) && slots[s].wiring.numInputs > 0 &&
                 falseOutputs.count(slots[s].wiring.inputSourceIndices[0]) != 0) {
        loop.exits.push_back(s);
      }
    }
    std::sort(loop.merges.begin(), loop.merges.end());
    std::sort(loop.switches.begin(), loop.switches.end());
    loops.push_back(loop);
  }
  std::sort(loops.begin(), loops.end(),
            [](const LoopSlots& a, const LoopSlots& b) { return a.merges.front() < b.merges.front(); });

  for (const LoopSlots& loop : loops) {
    LoopRegion region;
    region.mergeSlot = loop.merges.front();
    region.switchSlot = loop.switches.front();
    region.nextIterSlot = loop.nextIterations.back();
    region.exitSlot = loop.exits.empty() ? -1 : loop.exits.back();
    region.bodyStartSlot = loop.switches.back() + 1;
    region.bodyEndSlot = region.nextIterSlot;
    const std::string where = "the while loop at slot " + std::to_string(region.mergeSlot);
    for (int merge : loop.merges) {
      const auto& wiring = slots[merge].wiring;
      const int enter = wiring.numInputs == 2 ? producerOf(wiring.inputSourceIndices[0]) : -1;
      const int next = wiring.numInputs == 2 ? producerOf(wiring.inputSourceIndices[1]) : -1;
      if (!isType(enter, CF_ENTER) || enter >= region.mergeSlot || !isType(next, CF_NEXT_ITERATION)) {
        return fail(where + ": Merge '" + slots[merge].ident.opName +
                    "' must read an Enter placed before the loop and the loop's NextIteration");
      }
    }
    if (loop.merges.back() >= region.nextIterSlot || loop.switches.back() >= region.nextIterSlot ||
        loop.nextIterations.front() <= region.mergeSlot) {
      return fail(where + ": its Merges and Switches must all come before its last NextIteration (slot " +
                  std::to_string(region.nextIterSlot) + ") and its NextIterations after its first Merge");
    }
    regions->push_back(region);
  }

  // A pass runs from the first Merge to the last NextIteration; a restart reruns everything from
  // the first Merge on, so loops must nest or stay apart.
  for (size_t a = 0; a < regions->size(); a++) {
    for (size_t b = a + 1; b < regions->size(); b++) {
      const LoopRegion& outer = (*regions)[a];
      const LoopRegion& inner = (*regions)[b];
      if (inner.mergeSlot > outer.nextIterSlot) continue;
      if (inner.nextIterSlot > outer.nextIterSlot) {
        return fail("the while loops at slots " + std::to_string(outer.mergeSlot) + " and " +
                    std::to_string(inner.mergeSlot) + " overlap: a loop must lie inside another or apart");
      }
      if (loops[a].merges.back() > inner.mergeSlot) {
        return fail("the Merges of the while loop at slot " + std::to_string(outer.mergeSlot) +
                    " must all come before the loop nested at slot " + std::to_string(inner.mergeSlot));
      }
    }
  }

  // Outer loops first: a nested loop's block takes its own index. Merges and NextIterations
  // carry an index only as their own loop's (a Merge in the range may be an if's).
  for (int r = 0; r < static_cast<int>(regions->size()); r++) {
    const LoopRegion& region = (*regions)[r];
    for (int s = region.mergeSlot; s <= std::max(region.nextIterSlot, region.exitSlot); s++) {
      if (slots[s].cf.controlFlowType == CF_MERGE || slots[s].cf.controlFlowType == CF_NEXT_ITERATION) continue;
      slots[s].cf.loopRegionIndex = r;
    }
  }
  for (int r = 0; r < static_cast<int>(regions->size()); r++) {
    const LoopSlots& loop = loops[r];
    for (int s : loop.merges) slots[s].cf.loopRegionIndex = r;
    for (int s : loop.switches) slots[s].cf.loopRegionIndex = r;
    for (int s : loop.exits) slots[s].cf.loopRegionIndex = r;
    for (int s : loop.nextIterations) {
      slots[s].cf.loopRegionIndex = r;
      slots[s].cf.loopBackTarget = (*regions)[r].mergeSlot;
    }
  }
  return true;
}

void dspResetDeadFlags(const NativeSlot* slots, int numSlots, bool* slotIsDead, int slotIsDeadSize) {
  if (slotIsDead == nullptr) return;
  std::fill(slotIsDead, slotIsDead + slotIsDeadSize, false);
  for (int s = 0; s < numSlots; s++) {
    if (slots[s].cf.controlFlowType == CF_NEXT_ITERATION) setDead(slots[s], true, slotIsDead, slotIsDeadSize);
  }
}

DspSegmentLiveness dspSegmentLiveness(const NativeSlot* slots, int startSlot, int endSlot, bool* slotIsDead,
                                      int slotIsDeadSize) {
  if (slotIsDead == nullptr) return DspSegmentLiveness::ALL_LIVE;
  for (int s = startSlot; s <= endSlot; s++) {
    if (slots[s].cf.controlFlowType != CF_NONE) return DspSegmentLiveness::ALL_LIVE;
  }
  int dead = 0;
  for (int s = startSlot; s <= endSlot; s++) {
    const NativeSlot& slot = slots[s];
    bool readsDead = false;
    for (int i = 0; i < slot.wiring.numInputs && !readsDead; i++) {
      const int source = slot.wiring.inputSourceIndices[i];
      readsDead = source >= 0 && source < slotIsDeadSize && slotIsDead[source];
    }
    if (readsDead) {
      setDead(slot, true, slotIsDead, slotIsDeadSize);
      dead++;
    }
  }
  if (dead == 0) return DspSegmentLiveness::ALL_LIVE;
  return dead == endSlot - startSlot + 1 ? DspSegmentLiveness::ALL_DEAD : DspSegmentLiveness::MIXED;
}

bool dspLoopContinues(const NativeSlot* slots, const LoopRegion& region, const bool* slotIsDead,
                      int slotIsDeadSize) {
  // The Switch's true output is live exactly when the predicate held
  const NativeSlot& gate = slots[region.switchSlot];
  if (slotIsDead == nullptr || gate.wiring.numOutputs < 2) return false;
  const int trueOutput = gate.wiring.outputSlotIndices[1];
  return trueOutput >= 0 && trueOutput < slotIsDeadSize && !slotIsDead[trueOutput];
}

void dspPrepareLoopRestart(const NativeSlot* slots, int numSlots, const LoopRegion* regions, int numRegions,
                           int regionIndex, bool* slotIsDead, int slotIsDeadSize) {
  if (slotIsDead == nullptr || regionIndex < 0 || regionIndex >= numRegions) return;
  const int restart = regions[regionIndex].mergeSlot;
  for (int s = restart; s < numSlots; s++) {
    const NativeSlot& slot = slots[s];
    if (slot.cf.controlFlowType == CF_NEXT_ITERATION) {
      // The loop's own NextIterations hold the next pass's values; another loop in the range
      // starts over, with no next pass
      if (slot.cf.loopRegionIndex != regionIndex) setDead(slot, true, slotIsDead, slotIsDeadSize);
      continue;
    }
    // Runs again: the pass decides what is dead
    setDead(slot, false, slotIsDead, slotIsDeadSize);
  }
  for (int s = restart; s < numSlots; s++) {
    const NativeSlot& slot = slots[s];
    // A Merge with a loop index is that loop's (dspBuildLoopRegions)
    if (slot.cf.controlFlowType != CF_MERGE || slot.cf.loopRegionIndex < 0 || slot.wiring.numInputs < 2) continue;
    const int enter = slot.wiring.inputSourceIndices[0];
    if (enter < 0 || enter >= slotIsDeadSize) continue;
    // This loop's Merges take their NextIteration inputs; the others take their Enters again
    slotIsDead[enter] = slot.cf.loopRegionIndex == regionIndex;
  }
}

}  // namespace graph
}  // namespace sd
