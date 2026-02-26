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

#include <ops/declarable/helpers/fusedElementwiseChain.h>

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <cstring>

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
    "erfc", "log1p", "rsqrt",
    "relu6", "leakyrelu", "selu",
    "softplus", "softsign", "hard_sigmoid", "hardtanh",
    // Logical unary
    "boolean_not", "logical_not",
    // Identity/copy
    "identity", "assign",
    // Cast
    "cast",
};

static std::unordered_set<std::string> binaryElementwiseOps = {
    "add", "subtract", "multiply", "divide",
    "maximum", "minimum", "pow",
    "floormod", "floordiv",
    "atan2", "reversedivide", "reversesubtract",
    "squaredsubtract", "multiply_no_nan",
    "min_pairwise", "max_pairwise", "mod",
    "swish_mul",
    // Comparison ops
    "greater", "greater_equal", "less", "less_equal",
    "equals", "not_equals",
    // Logical binary
    "boolean_and", "boolean_or", "boolean_xor",
    "logical_and", "logical_or",
};

static std::unordered_set<std::string> ternaryElementwiseOps = {
    "where", "select",
};

static std::unordered_set<std::string> reductionOps = {
    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
    "reduce_norm1", "reduce_norm2", "reduce_logsumexp", "reduce_variance", "reduce_stdev",
    "sum", "mean", "max", "min", "prod", "norm1", "norm2", "normmax",
    "argmax", "argmin",
};

static std::unordered_set<std::string> normalizationOps = {
    "softmax", "log_softmax", "layer_norm", "batch_norm", "rms_norm",
    "normalize_moments",
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
        std::string name = *op->getOpName();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return name;
    }
    return "";
}

/**
 * Resolve op name directly from slot metadata. This avoids hash lookup for legacy
 * ops that are executable but not present in OpRegistrator hash map.
 */
static std::string getOpName(const NativeSlot& slot) {
    if (!slot.opName.empty()) {
        std::string name = slot.opName;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return name;
    }

    // Fallback for legacy plans that may not persist opName.
    return getOpName(slot.opHash);
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
 * Check if a slot is element-wise (unary, binary, or ternary).
 */
static bool isElementwiseSlot(const NativeSlot& slot) {
    std::string name = getOpName(slot);
    return unaryElementwiseOps.count(name) > 0
        || binaryElementwiseOps.count(name) > 0
        || ternaryElementwiseOps.count(name) > 0;
}

/**
 * Check if a slot is a reduction op.
 */
static bool isReductionSlot(const NativeSlot& slot) {
    std::string name = getOpName(slot);
    return reductionOps.count(name) > 0;
}

/**
 * Check if a slot is a normalization op.
 */
static bool isNormalizationSlot(const NativeSlot& slot) {
    std::string name = getOpName(slot);
    return normalizationOps.count(name) > 0;
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
    std::string bName = getOpName(slotB);
    bool bIsBinary = binaryElementwiseOps.count(bName) > 0;

    bool bIsTernary = ternaryElementwiseOps.count(bName) > 0;

    if (bIsTernary) {
        // Ternary op (where/select): needs exactly 3 inputs
        // At least one input must come from A's output
        if (slotB.numInputs != 3) return false;
        bool foundAOutput = false;
        for (int i = 0; i < slotB.numInputs; i++) {
            if (slotB.inputSourceIndices[i] == slotAOutputIdx) {
                foundAOutput = true;
                break;
            }
        }
        return foundAOutput;
    } else if (bIsBinary) {
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

        std::string opName = getOpName(slots[i]);
        if (opName != "add") continue;
        if (slots[i].numOutputs != 1) continue;

        int outputIdx = slots[i].outputSlotIndices[0];

        // Find activation consuming this add's output
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string nextName = getOpName(slots[j]);
            if (nextName.empty() || activationOps.count(nextName) == 0) continue;
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

    // Pass 3: Detect matmul → add(bias) → optional activation patterns
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;

        std::string opName = getOpName(slots[i]);
        if (opName != "matmul" && opName != "mmul"
            && opName != "fp8_matmul" && opName != "smooth_quant"
            && opName != "awq_matmul" && opName != "column_parallel_linear"
            && opName != "row_parallel_linear" && opName != "multi_lora_matmul"
            && opName != "fused_gemm_swiglu") continue;
        if (slots[i].numOutputs != 1) continue;

        int matmulOutputIdx = slots[i].outputSlotIndices[0];

        // Find add consuming matmul's output
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string addName = getOpName(slots[j]);
            if (addName != "add") continue;
            if (slots[j].numInputs != 2) continue;

            // One input must be matmul output, other must be external (bias)
            bool foundMatmulOutput = false;
            bool foundExternal = false;
            for (int k = 0; k < slots[j].numInputs; k++) {
                if (slots[j].inputSourceIndices[k] == matmulOutputIdx) foundMatmulOutput = true;
                else if (slots[j].inputSourceIndices[k] < 0) foundExternal = true;
            }
            if (!foundMatmulOutput || !foundExternal) continue;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, i)) continue;
            if (slots[j].numOutputs != 1) continue;

            // Found matmul → add. Now check for optional activation.
            std::vector<int> chain = {i, j};
            int addOutputIdx = slots[j].outputSlotIndices[0];

            for (int a = j + 1; a < numSlots; a++) {
                if (fused[a]) continue;
                std::string actName = getOpName(slots[a]);
                if (actName.empty() || activationOps.count(actName) == 0) continue;
                if (slots[a].numInputs != 1) continue;
                if (slots[a].inputSourceIndices[0] != addOutputIdx) continue;
                if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, j)) continue;

                chain.push_back(a);
                break;
            }

            // Create candidate if we have at least matmul + add (2 slots)
            if (chain.size() >= 2) {
                FusionCandidate candidate;
                candidate.startSlot = chain.front();
                candidate.endSlot = chain.back();
                candidate.type = FusionCandidate::MATMUL_BIAS_ACTIVATION;
                candidate.slotIndices = chain;
                candidate.chainLength = static_cast<int>(chain.size());
                candidates.push_back(candidate);

                for (int idx : chain) fused[idx] = true;
            }
            break;  // Only match first add per matmul
        }
    }

    // Pass 4: Detect reduction → element-wise chains (e.g., reduce_sum → sqrt for norm)
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isReductionSlot(slots[i])) continue;
        if (slots[i].numOutputs != 1) continue;

        int outputIdx = slots[i].outputSlotIndices[0];
        std::vector<int> chain;
        chain.push_back(i);

        // Look for element-wise ops consuming the reduction output
        for (int j = i + 1; j < numSlots && chain.size() < static_cast<size_t>(MAX_CHAIN_LENGTH); j++) {
            if (fused[j]) continue;
            if (!isElementwiseSlot(slots[j])) break;
            if (slots[j].numInputs < 1) break;

            bool consumesPrev = false;
            int prevOutputIdx = slots[chain.back()].outputSlotIndices[0];
            for (int k = 0; k < slots[j].numInputs; k++) {
                if (slots[j].inputSourceIndices[k] == prevOutputIdx) {
                    consumesPrev = true;
                    break;
                }
            }
            if (!consumesPrev) break;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, static_cast<int>(chain.back()))) break;
            if (slots[j].numOutputs != 1) break;

            chain.push_back(j);
        }

        if (chain.size() >= 2) {
            FusionCandidate candidate;
            candidate.startSlot = chain.front();
            candidate.endSlot = chain.back();
            candidate.type = FusionCandidate::ELEMENTWISE_CHAIN;
            candidate.slotIndices = chain;
            candidate.chainLength = static_cast<int>(chain.size());
            candidates.push_back(candidate);
            for (int idx : chain) fused[idx] = true;
        }
    }

    // Pass 5: Detect normalization → element-wise chains (e.g., softmax → log)
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isNormalizationSlot(slots[i])) continue;
        if (slots[i].numOutputs != 1) continue;

        std::vector<int> chain;
        chain.push_back(i);

        for (int j = i + 1; j < numSlots && chain.size() < static_cast<size_t>(MAX_CHAIN_LENGTH); j++) {
            if (fused[j]) continue;
            if (!isElementwiseSlot(slots[j])) break;
            if (slots[j].numInputs < 1) break;

            int prevOutputIdx = slots[chain.back()].outputSlotIndices[0];
            bool consumesPrev = false;
            for (int k = 0; k < slots[j].numInputs; k++) {
                if (slots[j].inputSourceIndices[k] == prevOutputIdx) {
                    consumesPrev = true;
                    break;
                }
            }
            if (!consumesPrev) break;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, static_cast<int>(chain.back()))) break;
            if (slots[j].numOutputs != 1) break;

            chain.push_back(j);
        }

        if (chain.size() >= 2) {
            FusionCandidate candidate;
            candidate.startSlot = chain.front();
            candidate.endSlot = chain.back();
            candidate.type = FusionCandidate::ELEMENTWISE_CHAIN;
            candidate.slotIndices = chain;
            candidate.chainLength = static_cast<int>(chain.size());
            candidates.push_back(candidate);
            for (int idx : chain) fused[idx] = true;
        }
    }

    // Pass 6: Detect softmax compound patterns (reduce_max → sub → exp → reduce_sum → div)
    for (int i = 0; i < numSlots - 4; i++) {
        if (fused[i]) continue;
        std::string op0 = getOpName(slots[i]);
        if (op0 != "reduce_max") continue;
        if (slots[i].numOutputs != 1) continue;

        int out0 = slots[i].outputSlotIndices[0];
        // Find sub consuming reduce_max output
        int subSlot = -1;
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "subtract") continue;
            for (int k = 0; k < slots[j].numInputs; k++) {
                if (slots[j].inputSourceIndices[k] == out0) { subSlot = j; break; }
            }
            if (subSlot >= 0) break;
        }
        if (subSlot < 0) continue;

        int outSub = slots[subSlot].outputSlotIndices[0];
        // Find exp consuming sub output
        int expSlot = -1;
        for (int j = subSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "exp") continue;
            if (slots[j].numInputs >= 1 && slots[j].inputSourceIndices[0] == outSub) {
                expSlot = j; break;
            }
        }
        if (expSlot < 0) continue;

        int outExp = slots[expSlot].outputSlotIndices[0];
        // Find reduce_sum consuming exp output
        int sumSlot = -1;
        for (int j = expSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "reduce_sum") continue;
            if (slots[j].numInputs >= 1 && slots[j].inputSourceIndices[0] == outExp) {
                sumSlot = j; break;
            }
        }
        if (sumSlot < 0) continue;

        int outSum = slots[sumSlot].outputSlotIndices[0];
        // Find div consuming exp and sum outputs
        int divSlot = -1;
        for (int j = sumSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "divide") continue;
            bool hasExp = false, hasSum = false;
            for (int k = 0; k < slots[j].numInputs; k++) {
                if (slots[j].inputSourceIndices[k] == outExp) hasExp = true;
                if (slots[j].inputSourceIndices[k] == outSum) hasSum = true;
            }
            if (hasExp && hasSum) { divSlot = j; break; }
        }
        if (divSlot < 0) continue;

        // Found softmax compound pattern!
        std::vector<int> chain = {i, subSlot, expSlot, sumSlot, divSlot};
        FusionCandidate candidate;
        candidate.startSlot = chain.front();
        candidate.endSlot = chain.back();
        candidate.type = FusionCandidate::ELEMENTWISE_CHAIN;
        candidate.slotIndices = chain;
        candidate.chainLength = static_cast<int>(chain.size());
        candidates.push_back(candidate);
        for (int idx : chain) fused[idx] = true;

        sd_printf("FusionPass: detected softmax compound pattern, slots %d-%d\n",
                  chain.front(), chain.back());
    }

    // Sort by startSlot for deterministic ordering
    std::sort(candidates.begin(), candidates.end(),
              [](const FusionCandidate& a, const FusionCandidate& b) {
                  return a.startSlot < b.startSlot;
              });

    return candidates;
}

int FusionPass::applyFusions(
        NativeSlot* slots, int numSlots,
        const std::vector<FusionCandidate>& candidates) {

    int applied = 0;

    for (const auto& fusion : candidates) {
        switch (fusion.type) {

            case FusionCandidate::ELEMENTWISE_CHAIN: {
                // For a chain A → B → C → ...:
                // Each op after A reuses its primary input buffer as its output.
                // This eliminates intermediate allocations — the chain runs in-place
                // through a single buffer.
                //
                // The first op (A) allocates normally. B writes its output into A's
                // output buffer. C writes into B's (=A's) output buffer. Etc.
                if (fusion.slotIndices.size() < 2) break;

                for (size_t i = 1; i < fusion.slotIndices.size(); i++) {
                    int slotIdx = fusion.slotIndices[i];
                    if (slotIdx < 0 || slotIdx >= numSlots) continue;

                    NativeSlot& slot = slots[slotIdx];

                    // Find which input comes from the previous op in the chain
                    int prevSlotIdx = fusion.slotIndices[i - 1];
                    int prevOutputSlot = slots[prevSlotIdx].outputSlotIndices[0];

                    int fusedInputIdx = -1;
                    for (int k = 0; k < slot.numInputs; k++) {
                        if (slot.inputSourceIndices[k] == prevOutputSlot) {
                            fusedInputIdx = k;
                            break;
                        }
                    }

                    if (fusedInputIdx >= 0) {
                        slot.inPlaceFused = true;
                        slot.inPlaceFusedInputIdx = fusedInputIdx;
                    }
                }

                // ── Fused kernel dispatch metadata ──────────────────────────
                // Try to map all ops in the chain to FusedElemOp codes.
                // If all succeed, mark head for fused kernel dispatch and tails for skip.
                {
                    int chainLen = static_cast<int>(fusion.slotIndices.size());
                    if (chainLen <= 8) {
                        bool allMapped = true;
                        int opCodes[8] = {};
                        int secondarySources[8] = {};

                        for (int ci = 0; ci < chainLen; ci++) {
                            int si = fusion.slotIndices[ci];
                            std::string name = getOpName(slots[si]);
                            int code = sd::ops::helpers::opNameToFusedCode(name);
                            if (code < 0) {
                                allMapped = false;
                                break;
                            }
                            opCodes[ci] = code;

                            // For binary ops, find the secondary input source
                            // (the one that does NOT come from the previous chain slot)
                            secondarySources[ci] = INT32_MIN;  // INT32_MIN = unary, no secondary (-1 is external input 0!)
                            if (binaryElementwiseOps.count(name) > 0 && slots[si].numInputs == 2) {
                                int prevOutputSlotIdx = -1;
                                if (ci > 0) {
                                    prevOutputSlotIdx = slots[fusion.slotIndices[ci - 1]].outputSlotIndices[0];
                                } else {
                                    // Head slot: primary input is inputSourceIndices[0]
                                    // Secondary is whatever is NOT that
                                    // (for head binary op, both inputs are "external" to the chain)
                                    // The chain input is input 0, secondary is input 1
                                    // But actually the head's primary chain input could be either index
                                }

                                for (int k = 0; k < slots[si].numInputs; k++) {
                                    int srcIdx = slots[si].inputSourceIndices[k];
                                    if (ci == 0) {
                                        // Head binary op: secondary is the external input (srcIdx < 0)
                                        if (srcIdx < 0) {
                                            secondarySources[ci] = srcIdx;
                                            break;
                                        }
                                    } else {
                                        // Non-head: secondary is whichever input is NOT from prev chain slot
                                        if (srcIdx != prevOutputSlotIdx) {
                                            secondarySources[ci] = srcIdx;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if (allMapped) {
                            // Populate head slot
                            int headIdx = fusion.slotIndices[0];
                            NativeSlot& headSlot = slots[headIdx];
                            headSlot.isFusedChainHead = true;
                            headSlot.fusedChainLength = chainLen;
                            std::memcpy(headSlot.fusedChainOpCodes, opCodes, sizeof(int) * chainLen);
                            std::memcpy(headSlot.fusedChainSecondaryInputSources, secondarySources, sizeof(int) * chainLen);
                            for (int ci = 0; ci < chainLen; ci++) {
                                headSlot.fusedChainSlots[ci] = fusion.slotIndices[ci];
                            }

                            // Mark tail slots (all except head)
                            for (int ci = 1; ci < chainLen; ci++) {
                                slots[fusion.slotIndices[ci]].isFusedChainTail = true;
                            }

                            sd_printf("FusionPass: fused kernel dispatch enabled for chain slots %d-%d (%d ops)\n",
                                      headIdx, fusion.slotIndices.back(), chainLen);
                        }
                    }
                }

                applied++;
                sd_printf("FusionPass: applied ELEMENTWISE_CHAIN fusion, slots %d-%d (%d ops)\n",
                          fusion.startSlot, fusion.endSlot, fusion.chainLength);
                break;
            }

            case FusionCandidate::BIAS_ACTIVATION: {
                // add(x, bias) → activation(result)
                // The activation op reuses the add's output buffer.
                if (fusion.slotIndices.size() != 2) break;

                int addSlotIdx = fusion.slotIndices[0];
                int actSlotIdx = fusion.slotIndices[1];
                if (actSlotIdx < 0 || actSlotIdx >= numSlots) break;
                if (addSlotIdx < 0 || addSlotIdx >= numSlots) break;

                NativeSlot& actSlot = slots[actSlotIdx];
                int addOutputSlot = slots[addSlotIdx].outputSlotIndices[0];

                // Find which input of the activation comes from the add
                int fusedInputIdx = -1;
                for (int k = 0; k < actSlot.numInputs; k++) {
                    if (actSlot.inputSourceIndices[k] == addOutputSlot) {
                        fusedInputIdx = k;
                        break;
                    }
                }

                if (fusedInputIdx >= 0) {
                    actSlot.inPlaceFused = true;
                    actSlot.inPlaceFusedInputIdx = fusedInputIdx;
                    applied++;
                    sd_printf("FusionPass: applied BIAS_ACTIVATION fusion, slots %d-%d\n",
                              fusion.startSlot, fusion.endSlot);
                }
                break;
            }

            case FusionCandidate::MATMUL_BIAS_ACTIVATION: {
                // matmul → add(bias) → activation
                // The add and activation ops run in-place on the matmul's output.
                if (fusion.slotIndices.size() < 2) break;

                // Apply in-place starting from the second op (add)
                for (size_t i = 1; i < fusion.slotIndices.size(); i++) {
                    int slotIdx = fusion.slotIndices[i];
                    if (slotIdx < 0 || slotIdx >= numSlots) continue;

                    NativeSlot& slot = slots[slotIdx];
                    int prevSlotIdx = fusion.slotIndices[i - 1];
                    int prevOutputSlot = slots[prevSlotIdx].outputSlotIndices[0];

                    int fusedInputIdx = -1;
                    for (int k = 0; k < slot.numInputs; k++) {
                        if (slot.inputSourceIndices[k] == prevOutputSlot) {
                            fusedInputIdx = k;
                            break;
                        }
                    }

                    if (fusedInputIdx >= 0) {
                        slot.inPlaceFused = true;
                        slot.inPlaceFusedInputIdx = fusedInputIdx;
                    }
                }

                applied++;
                sd_printf("FusionPass: applied MATMUL_BIAS_ACTIVATION fusion, slots %d-%d\n",
                          fusion.startSlot, fusion.endSlot);
                break;
            }
        }
    }

    return applied;
}

}  // namespace graph
}  // namespace sd
