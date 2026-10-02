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

#ifndef LIBND4J_DSP_LOOP_STRUCTURE_H
#define LIBND4J_DSP_LOOP_STRUCTURE_H

#include <graph/NativeDynamicShapePlan.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Derives a plan's while loops from its wiring and checks that the plan can run them.
 *
 * A loop is the Merges fed by a NextIteration whose Switches share one predicate. Its region runs
 * from its first Merge (mergeSlot) to its last NextIteration (nextIterSlot); switchSlot is its
 * first Switch and exitSlot its last Exit. Regions come in mergeSlot order. Each of the loop's
 * NextIterations gets loopBackTarget = mergeSlot. Every slot from mergeSlot to the loop's last
 * NextIteration or Exit gets loopRegionIndex = the loop's index, a nested loop's slots its own;
 * the loop's Merges, Switches, NextIterations and Exits always carry their loop's index.
 *
 * A pass of a loop runs from its first Merge to its last NextIteration, where the plan restarts it
 * (dspPrepareLoopRestart) while its predicate holds. That needs:
 *   - each Merge to read an Enter placed before the loop (input 0) and the loop's NextIteration
 *     (input 1);
 *   - every Merge, Switch and NextIteration of the loop between its first Merge and its last
 *     NextIteration;
 *   - two loops' regions apart or one inside the other, with the outer loop's Merges all ahead
 *     of the inner loop.
 * Returns false with *error naming the first violation.
 */
SD_LIB_HIDDEN bool dspBuildLoopRegions(NativeSlot* slots, int numSlots, int totalOutputSlots,
                                       std::vector<LoopRegion>* regions, std::string* error);

/**
 * The dead flags an execution starts from: every output live but the NextIterations', since no
 * loop has a next pass yet. phaseReplay and phaseWarmup, which every execution runs one of, start
 * from them; an execution must not inherit the last one's loop state.
 */
SD_LIB_HIDDEN void dspResetDeadFlags(const NativeSlot* slots, int numSlots, bool* slotIsDead,
                                     int slotIsDeadSize);

/** Which ops of a segment run in the current pass of a plan with control flow. */
enum class DspSegmentLiveness { ALL_LIVE, ALL_DEAD, MIXED };

/**
 * Decides which ops of segment [startSlot, endSlot] are dead in the current pass, as
 * executeSegmentSlotBySlot would (an op reading a dead value is dead), and marks their outputs dead.
 * A loop body is dead in the pass that ends the loop. A captured graph or compiled kernel would run
 * all of the segment's ops, so dispatchSegment skips an ALL_DEAD segment and runs a MIXED one slot
 * by slot. A segment holding control-flow ops is ALL_LIVE here: its slot-by-slot dispatch decides.
 */
SD_LIB_HIDDEN DspSegmentLiveness dspSegmentLiveness(const NativeSlot* slots, int startSlot, int endSlot,
                                                    bool* slotIsDead, int slotIsDeadSize);

/** Whether a loop's predicate held in the pass that just reached its last NextIteration. */
SD_LIB_HIDDEN bool dspLoopContinues(const NativeSlot* slots, const LoopRegion& region,
                                    const bool* slotIsDead, int slotIsDeadSize);

/**
 * Sets the dead flags for another pass of loop regionIndex, which restarts at its first Merge.
 * Every slot from there on runs again. The loop's Merges take their NextIteration inputs. Every
 * other loop starting in that range starts over from its Enter inputs, with no next pass.
 */
SD_LIB_HIDDEN void dspPrepareLoopRestart(const NativeSlot* slots, int numSlots,
                                         const LoopRegion* regions, int numRegions, int regionIndex,
                                         bool* slotIsDead, int slotIsDeadSize);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_LOOP_STRUCTURE_H
