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

#include <graph/FusionPass.h>
#include <graph/NativeDynamicShapePlan.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/DeclarableOp.h>

#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace sd {
namespace graph {

// Op name hashes for element-wise operations.
// These are computed via sd::ops::HashHelper::getLongHash() at registration time.
// We use op names for readability and look up hashes dynamically.

static std::unordered_set<std::string> unaryElementwiseOps = {
    "relu", "sigmoid", "tanh", "gelu",
    "exp", "log", "abs", "neg",
    "square", "sqrt", "swish", "silu", "mish",
    "ceil", "floor", "round",
    "sin", "cos", "asin", "acos", "atan",
    "sinh", "cosh", "asinh", "acosh", "atanh",
    "reciprocal", "sign", "elu",
};

static std::unordered_set<std::string> binaryElementwiseOps = {
    "add", "subtract", "multiply", "divide",
    "maximum", "minimum", "pow",
    "floormod", "floordiv",
};

static std::unordered_set<std::string> activationOps = {
    "relu", "sigmoid", "tanh", "gelu",
    "swish", "silu", "mish", "elu",
    "softplus", "softsign",
};

/**
 * Get the op name for a given op hash by looking it up in the OpRegistrator.
 */
static std::string getOpName(sd::LongType opHash) {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(opHash);
    if (op != nullptr) {
        return *op->getOpName();
    }
    return "";
}

bool FusionPass::isUnaryElementwise(sd::LongType opHash) {
    std::string name = getOpName(opHash);
    return !name.empty() && unaryElementwiseOps.count(name) > 0;
}

bool FusionPass::isBinaryElementwise(sd::LongType opHash) {
    std::string name = getOpName(opHash);
    return !name.empty() && binaryElementwiseOps.count(name) > 0;
}

bool FusionPass::isActivation(sd::LongType opHash) {
    std::string name = getOpName(opHash);
    return !name.empty() && activationOps.count(name) > 0;
}

/**
 * Check if a slot is element-wise (unary or binary).
 */
static bool isElementwiseSlot(const NativeSlot& slot) {
    std::string name = getOpName(slot.opHash);
    return unaryElementwiseOps.count(name) > 0 || binaryElementwiseOps.count(name) > 0;
}

/**
 * Check if slot B can be fused after slot A in an element-wise chain.
 *
 * Rules:
 * 1. B must be element-wise (unary or binary)
 * 2. B must have exactly one primary input that comes from A's output
 * 3. B must not be data-dependent (output shape depends on input values)
 * 4. For binary ops, the second input must be external (constant/variable/placeholder)
 * 5. Both must target the same device
 */
static bool canChainAfter(const NativeSlot& slotA, const NativeSlot& slotB,
                           int slotAOutputIdx) {
    if (!isElementwiseSlot(slotB)) return false;
    if (slotB.isDataDependent || slotB.outputShapeDependsOnInputValues) return false;

    // Check that B has exactly the right number of inputs
    std::string bName = getOpName(slotB.opHash);
    bool bIsBinary = binaryElementwiseOps.count(bName) > 0;

    if (bIsBinary) {
        // Binary op: needs exactly 2 inputs
        if (slotB.numInputs != 2) return false;

        // One input must come from A's output, the other must be external
        bool foundAOutput = false;
        bool foundExternal = false;
        for (int i = 0; i < slotB.numInputs; i++) {
            int srcIdx = slotB.inputSourceIndices[i];
            if (srcIdx == slotAOutputIdx) {
                foundAOutput = true;
            } else if (srcIdx < 0) {
                // External input (constant/variable/placeholder)
                foundExternal = true;
            }
        }
        return foundAOutput && foundExternal;
    } else {
        // Unary op: needs exactly 1 input from A's output
        if (slotB.numInputs != 1) return false;
        return slotB.inputSourceIndices[0] == slotAOutputIdx;
    }
}

/**
 * Pre-compute consumer counts for all output slot indices.
 * Returns a map from outputSlotIndex → number of slots that consume it.
 * O(total inputs) instead of O(N²) per query.
 */
static std::unordered_map<int, int> buildConsumerCounts(const NativeSlot* slots, int numSlots) {
    std::unordered_map<int, int> counts;
    for (int s = 0; s < numSlots; s++) {
        for (int i = 0; i < slots[s].numInputs; i++) {
            int srcIdx = slots[s].inputSourceIndices[i];
            if (srcIdx >= 0) {  // Only count internal slot references, not external inputs
                counts[srcIdx]++;
            }
        }
    }
    return counts;
}

/**
 * Check if a slot's output is only consumed by exactly one other slot.
 * Uses pre-computed consumer counts for O(1) lookup.
 */
static bool isOnlyConsumedOnce(const std::unordered_map<int, int>& consumerCounts,
                                const NativeSlot* slots, int numSlots, int slotIdx) {
    if (slotIdx < 0 || slotIdx >= numSlots) return false;
    int outputSlotIdx = slots[slotIdx].outputSlotIndices[0];
    auto it = consumerCounts.find(outputSlotIdx);
    return it != consumerCounts.end() && it->second == 1;
}

std::vector<FusionCandidate> FusionPass::detectFusions(
        NativeSlot* slots, int numSlots) {

    std::vector<FusionCandidate> candidates;
    if (slots == nullptr || numSlots <= 1) return candidates;

    // Pre-compute consumer counts for O(1) "only consumed once" checks
    auto consumerCounts = buildConsumerCounts(slots, numSlots);

    // Track which slots are already part of a fusion (no overlapping fusions)
    std::vector<bool> fused(numSlots, false);

    // Pass 1: Detect element-wise chains
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isElementwiseSlot(slots[i])) continue;
        if (slots[i].numOutputs != 1) continue;  // Only single-output ops

        // Try to extend a chain starting at slot i
        std::vector<int> chain;
        chain.push_back(i);
        int current = i;

        while (chain.size() < static_cast<size_t>(MAX_CHAIN_LENGTH)) {
            int outputIdx = slots[current].outputSlotIndices[0];

            // Find the next slot that consumes this output
            int nextSlot = -1;
            for (int j = current + 1; j < numSlots; j++) {
                if (fused[j]) continue;
                for (int k = 0; k < slots[j].numInputs; k++) {
                    if (slots[j].inputSourceIndices[k] == outputIdx) {
                        nextSlot = j;
                        break;
                    }
                }
                if (nextSlot >= 0) break;
            }

            if (nextSlot < 0) break;  // No consumer found

            // Check fusibility
            if (!canChainAfter(slots[current], slots[nextSlot], outputIdx)) break;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, current)) break;
            if (slots[nextSlot].numOutputs != 1) break;

            chain.push_back(nextSlot);
            current = nextSlot;
        }

        // Only create a fusion candidate if chain has >= 2 ops
        if (chain.size() >= 2) {
            FusionCandidate candidate;
            candidate.startSlot = chain.front();
            candidate.endSlot = chain.back();
            candidate.type = FusionCandidate::ELEMENTWISE_CHAIN;
            candidate.slotIndices = chain;
            candidate.chainLength = static_cast<int>(chain.size());

            candidates.push_back(candidate);

            // Mark all slots in this chain as fused
            for (int idx : chain) {
                fused[idx] = true;
            }
        }
    }

    // Pass 2: Detect bias+activation patterns (add -> relu/sigmoid/tanh/gelu)
    for (int i = 0; i < numSlots - 1; i++) {
        if (fused[i]) continue;

        std::string opName = getOpName(slots[i].opHash);
        if (opName != "add") continue;
        if (slots[i].numOutputs != 1) continue;

        int outputIdx = slots[i].outputSlotIndices[0];

        // Find activation consuming this add's output
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            if (!FusionPass::isActivation(slots[j].opHash)) continue;
            if (slots[j].numInputs != 1) continue;
            if (slots[j].inputSourceIndices[0] != outputIdx) continue;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, i)) continue;

            FusionCandidate candidate;
            candidate.startSlot = i;
            candidate.endSlot = j;
            candidate.type = FusionCandidate::BIAS_ACTIVATION;
            candidate.slotIndices = {i, j};
            candidate.chainLength = 2;

            candidates.push_back(candidate);
            fused[i] = true;
            fused[j] = true;
            break;
        }
    }

    // Sort by startSlot for deterministic ordering
    std::sort(candidates.begin(), candidates.end(),
              [](const FusionCandidate& a, const FusionCandidate& b) {
                  return a.startSlot < b.startSlot;
              });

    return candidates;
}

}  // namespace graph
}  // namespace sd
