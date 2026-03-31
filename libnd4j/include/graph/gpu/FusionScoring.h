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

#ifndef LIBND4J_FUSION_SCORING_H
#define LIBND4J_FUSION_SCORING_H

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/NDArray.h>
#include <vector>

namespace sd {
namespace graph {

/**
 * Score the benefit of extending an existing compile range by one adjacent section.
 *
 * Returns a float score:
 *   Negative = do not merge (grid incompatibility, shared mem overflow)
 *   Positive = merge is beneficial (higher = more beneficial)
 *
 * Factors:
 *   - Grid-type compatibility, with an explicit exception for attention neighborhoods
 *   - Intermediate memory traffic eliminated (bytes saved)
 *   - Kernel launch overhead saved (~15μs per eliminated launch)
 *   - Register pressure penalty for very large fused kernels
 *   - Maximum per-section shared memory must fit SM limit
 */
float scoreSectionFusionRange(
    const std::vector<KernelSection>& sections,
    int rangeStartSectionIdx,
    int rangeEndSectionIdx,
    int nextSectionIdx,
    NativeSlot* slots,
    int segmentStartSlot,
    int segmentEndSlot,
    NDArray** outputSlots,
    int totalOutputSlots);

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_FUSION_SCORING_H
