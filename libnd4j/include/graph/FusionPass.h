/* ******************************************************************************
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
// @author Adam Gibson
//
// Automatic op fusion pass for NativeDynamicShapePlan.
// Detects fusible patterns in the execution plan and replaces multi-op
// sequences with single fused kernels that eliminate intermediate buffers
// and kernel launch overhead.
//

#ifndef LIBND4J_FUSION_PASS_H
#define LIBND4J_FUSION_PASS_H

#include <system/common.h>
#include <climits>
#include <vector>
#include <string>

namespace sd {
namespace graph {

// Forward declaration
struct NativeSlot;

/**
 * Describes a detected fusible pattern in the execution plan.
 */
struct FusionCandidate {
    int startSlot;              // First slot in the fusible sequence
    int endSlot;                // Last slot (inclusive)

    enum FusionType {
        ELEMENTWISE_CHAIN,      // Consecutive unary/binary elementwise ops
        BIAS_ACTIVATION,        // add(bias) -> activation (relu/sigmoid/tanh/gelu)
        MATMUL_BIAS_ACTIVATION, // matmul -> add(bias) -> activation
    };

    FusionType type;
    std::vector<int> slotIndices;   // All slots consumed by this fusion
    int chainLength;                // Number of ops in the chain
};

/**
 * Fusion pass that analyzes and transforms execution plans.
 *
 * The pass runs during plan compilation and identifies multi-op sequences
 * that can be replaced with single fused kernels. This eliminates:
 * - Intermediate buffer allocations (O(N) -> O(1) for chain of N ops)
 * - Kernel launch overhead (one launch instead of N)
 * - Global memory round-trips (data stays in registers/L1)
 */
class SD_LIB_EXPORT FusionPass {
public:
    /**
     * Analyze a plan and return all detected fusion candidates.
     *
     * @param slots Array of NativeSlot describing each op
     * @param numSlots Number of slots in the plan
     * @param externalInputRanks Optional ranks for external inputs (size = numExternalInputs).
     *                           Used by MATMUL_BIAS_ACTIVATION fusion to validate that the
     *                           bias operand is a 1D vector. Pass empty vector if unknown
     *                           (some MATMUL_BIAS_ACTIVATION candidates will be skipped).
     * @return Vector of fusion candidates, sorted by startSlot
     */
    static std::vector<FusionCandidate> detectFusions(
        NativeSlot* slots, int numSlots,
        const std::vector<int>& externalInputRanks = {});

    /**
     * Apply detected fusions by marking slots for in-place execution.
     *
     * For ELEMENTWISE_CHAIN: Each op after the first reuses the previous
     * op's output buffer, eliminating intermediate allocations.
     *
     * For BIAS_ACTIVATION: The activation op reuses the add op's output buffer.
     *
     * @param slots Array of NativeSlot (modified in place)
     * @param numSlots Number of slots
     * @param candidates Fusion candidates from detectFusions()
     * @return Number of fusions successfully applied
     */
    static int applyFusions(
        NativeSlot* slots, int numSlots,
        const std::vector<FusionCandidate>& candidates);

    /**
     * Check if an op hash corresponds to a unary elementwise operation.
     */
    static bool isUnaryElementwise(sd::LongType opHash);

    /**
     * Check if an op hash corresponds to a binary elementwise operation.
     */
    static bool isBinaryElementwise(sd::LongType opHash);

    /**
     * Check if an op hash corresponds to an activation function.
     */
    static bool isActivation(sd::LongType opHash);

    /**
     * Maximum chain length for element-wise in-place buffer chaining.
     * No practical limit — each op is still a separate kernel launch,
     * so chain length only affects buffer reuse (no register pressure).
     */
    static constexpr int MAX_CHAIN_LENGTH = INT_MAX;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FUSION_PASS_H
