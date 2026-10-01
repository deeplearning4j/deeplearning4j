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
#include <array/ArrayOptions.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/LegacyOpTypeCodes.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/OpDescriptor.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>

#include <ops/declarable/helpers/fusedElementwiseChain.h>

#include <climits>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <cstring>

namespace sd {
namespace graph {

// Op classification is driven by each resolved op's OpDescriptor traits.

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
 * Resolve op name directly from slot metadata.
 */
static std::string getOpName(const NativeSlot& slot) {
    if (!slot.ident.opName.empty()) {
        std::string name = slot.ident.opName;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return name;
    }
    return getOpName(slot.ident.opHash);
}

/**
 * Query traits from a slot's resolved op descriptor.
 */
static bool slotHasTrait(const NativeSlot& slot, uint32_t trait) {
    if (slot.ident.op && slot.ident.op->getOpDescriptor()) {
        return slot.ident.op->getOpDescriptor()->hasAnyTrait(trait);
    }
    return false;
}

// MATMUL describes semantics, not an MmulHelper/cuBLASLt operand ABI.
// Packed/checkpoint linear ops also have the trait but cannot consume a sunk
// activation cast (it changes weight rounding) or the dense Lt epilogue state.
static bool isDenseMmulSlot(const NativeSlot& slot) {
    const auto name = getOpName(slot);
    return slotHasTrait(slot, sd::ops::OP_TRAIT_MATMUL) &&
           (name == "matmul" || name == "mmul") &&
           slot.wiring.numInputs == 2 && slot.wiring.numOutputs == 1 &&
           (slot.args.numIArgs < 4 || slot.args.iArgs[3] == 0);
}

// Explicit representability relation, not dtype enum ordering or a float category
// test: HALF and BFLOAT16 cannot represent every value of each other.
static bool isLosslessFloatWidening(DataType source, DataType target) {
    return ((source == HALF || source == BFLOAT16) &&
            (target == FLOAT32 || target == DOUBLE)) ||
           (source == FLOAT32 && target == DOUBLE);
}

// Read only retained shape metadata or the cast's declared target. Never inspect
// saved NDArray pointers: warmup arrays may already have been retired. External
// input dtypes are not part of detectFusions' ABI and must remain UNKNOWN.
static DataType retainedOutputType(const NativeSlot& slot, int output) {
    if (slotHasTrait(slot, sd::ops::OP_TRAIT_CAST) &&
        slot.wiring.numOutputs == 1 && slot.args.numIArgs > 0) {
        return static_cast<DataType>(slot.args.iArgs[0]);
    }
    if (slot.shapeCacheValid() &&
        output < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) &&
        slot.shapeCache.cachedOutputShapes[output] != nullptr) {
        return ArrayOptions::dataType(slot.shapeCache.cachedOutputShapes[output]);
    }
    if (output < static_cast<int>(slot.shapeCache.staticOutputShapeInfos.size()) &&
        !slot.shapeCache.staticOutputShapeInfos[output].empty()) {
        return ArrayOptions::dataType(slot.shapeCache.staticOutputShapeInfos[output].data());
    }
    return DataType::UNKNOWN;
}

static bool isElementwiseSlot(const NativeSlot& slot) {
    return slotHasTrait(slot, sd::ops::OP_TRAIT_UNARY_ELEMENTWISE |
                              sd::ops::OP_TRAIT_BINARY_ELEMENTWISE |
                              sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE);
}

static bool isReductionSlot(const NativeSlot& slot) {
    return slotHasTrait(slot, sd::ops::OP_TRAIT_REDUCTION);
}

static bool isNormalizationSlot(const NativeSlot& slot) {
    return slotHasTrait(slot, sd::ops::OP_TRAIT_NORMALIZATION);
}

/**
 * Check if slot B can be fused after slot A in an element-wise chain.
 *
 * Rules:
 * 1. B must be element-wise (unary or binary)
 * 2. B must have exactly one primary input that comes from A's output
 * 3. B must not be data-dependent (output shape depends on input values)
 * 4. For binary ops, the second input must be external (constant/variable/placeholder)
 */
static bool canChainAfter(const NativeSlot& slotA, const NativeSlot& slotB,
                           int slotAOutputIdx) {
    if (!isElementwiseSlot(slotB)) return false;
    if (slotB.hasValueDependentShape()) return false;

    // Check that B has exactly the right number of inputs
    bool bIsBinary = slotHasTrait(slotB, sd::ops::OP_TRAIT_BINARY_ELEMENTWISE);
    bool bIsTernary = slotHasTrait(slotB, sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE);

    if (bIsTernary) {
        // Ternary op (where/select): needs exactly 3 inputs
        // At least one input must come from A's output
        if (slotB.wiring.numInputs != 3) return false;
        bool foundAOutput = false;
        for (int i = 0; i < slotB.wiring.numInputs; i++) {
            if (slotB.wiring.inputSourceIndices[i] == slotAOutputIdx) {
                foundAOutput = true;
                break;
            }
        }
        return foundAOutput;
    } else if (bIsBinary) {
        // Binary op: needs exactly 2 inputs
        if (slotB.wiring.numInputs != 2) return false;

        // One input must come from A's output, the other must be external
        bool foundAOutput = false;
        bool foundExternal = false;
        for (int i = 0; i < slotB.wiring.numInputs; i++) {
            int srcIdx = slotB.wiring.inputSourceIndices[i];
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
        if (slotB.wiring.numInputs != 1) return false;
        return slotB.wiring.inputSourceIndices[0] == slotAOutputIdx;
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
        for (int i = 0; i < slots[s].wiring.numInputs; i++) {
            int srcIdx = slots[s].wiring.inputSourceIndices[i];
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
    int outputSlotIdx = slots[slotIdx].wiring.outputSlotIndices[0];
    auto it = consumerCounts.find(outputSlotIdx);
    return it != consumerCounts.end() && it->second == 1;
}

/**
 * Extend a chain starting from an already-identified head slot by appending
 * consecutive element-wise ops that consume the previous slot's output.
 * Used by reduction→elementwise and normalization→elementwise passes.
 *
 * @param startSlot     Index of the head slot (already in chain)
 * @param slots         Slot array
 * @param numSlots      Total slot count
 * @param fused         Per-slot fused flags (skips already-fused slots)
 * @param consumerCounts Pre-computed consumer counts
 * @return Chain of slot indices (including startSlot). Empty if no extension found.
 */
static std::vector<int> extendElementwiseChain(
        int startSlot,
        const NativeSlot* slots, int numSlots,
        const std::vector<bool>& fused,
        const std::unordered_map<int, int>& consumerCounts) {

    std::vector<int> chain;
    chain.push_back(startSlot);

    for (int j = startSlot + 1; j < numSlots && chain.size() < static_cast<size_t>(FusionPass::MAX_CHAIN_LENGTH); j++) {
        if (fused[j]) continue;
        if (!isElementwiseSlot(slots[j])) break;
        if (slots[j].wiring.numInputs < 1) break;

        int prevOutputIdx = slots[chain.back()].wiring.outputSlotIndices[0];
        bool consumesPrev = false;
        for (int k = 0; k < slots[j].wiring.numInputs; k++) {
            if (slots[j].wiring.inputSourceIndices[k] == prevOutputIdx) {
                consumesPrev = true;
                break;
            }
        }
        if (!consumesPrev) break;
        if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, static_cast<int>(chain.back()))) break;
        if (slots[j].wiring.numOutputs != 1) break;

        chain.push_back(j);
    }

    return chain;
}

/**
 * Pass 2: Reduction epilogue fusion.
 * Detects: reduce_* → elementwise chain
 * Fuses the elementwise tail into the reduction's epilogue.
 * Complements Pass 6 by adding isOnlyConsumedOnce check and isFusedChainTail marking.
 */
static void fuseReductionEpilogues(NativeSlot* slots, int numSlots,
                                    std::vector<bool>& fused,
                                    const std::unordered_map<int, int>& consumerCounts) {
    for (int s = 0; s < numSlots; s++) {
        if (fused[s]) continue;
        if (!isReductionSlot(slots[s])) continue;
        if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, s)) continue;

        auto chain = extendElementwiseChain(s, slots, numSlots, fused, consumerCounts);
        if (chain.size() > 1) {
            // Mark chain members as fused (but NOT as isFusedChainTail — no
            // corresponding head is set up, so tail-marking would skip execution).
            for (size_t i = 1; i < chain.size(); i++) {
                fused[chain[i]] = true;
            }
            DSP_DIAG(FUSION, "REDUCTION_EPILOGUE: slots [%d] + %zu elementwise tail ops (reserved, not tail-marked)",
                     s, chain.size() - 1);
        }
    }
}

/**
 * Pass 3: Normalization epilogue fusion.
 * Detects: layer_norm/rms_norm/softmax → elementwise chain
 * Fuses post-norm scaling/bias into the normalization kernel.
 * Complements Pass 7 by adding isOnlyConsumedOnce check and isFusedChainTail marking.
 */
static void fuseNormalizationEpilogues(NativeSlot* slots, int numSlots,
                                        std::vector<bool>& fused,
                                        const std::unordered_map<int, int>& consumerCounts) {
    for (int s = 0; s < numSlots; s++) {
        if (fused[s]) continue;
        if (!isNormalizationSlot(slots[s])) continue;
        if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, s)) continue;

        auto chain = extendElementwiseChain(s, slots, numSlots, fused, consumerCounts);
        if (chain.size() > 1) {
            // Mark chain members as fused (but NOT as isFusedChainTail — no
            // corresponding head is set up, so tail-marking would skip execution).
            for (size_t i = 1; i < chain.size(); i++) {
                fused[chain[i]] = true;
            }
            DSP_DIAG(FUSION, "NORM_EPILOGUE: slots [%d] + %zu elementwise tail ops (reserved, not tail-marked)",
                     s, chain.size() - 1);
        }
    }
}

std::vector<FusionCandidate> FusionPass::detectFusions(
        NativeSlot* slots, int numSlots,
        const std::vector<int>& externalInputRanks,
        const int* requestedOutputSlots, int numRequestedOutputs) {

    std::vector<FusionCandidate> candidates;
    if (slots == nullptr || numSlots <= 1) {
        DSP_DIAG(FUSION, "detectFusions: skipped (slots=%p numSlots=%d)", slots, numSlots);
        return candidates;
    }

    DSP_DIAG(FUSION, "detectFusions: BEGIN numSlots=%d castElim=%d castSink=%d",
             numSlots,
             Environment::getInstance().dspCastElimination() ? 1 : 0,
             Environment::getInstance().dspCastSinkMatmul() ? 1 : 0);

    // Pre-compute consumer counts for O(1) "only consumed once" checks
    auto consumerCounts = buildConsumerCounts(slots, numSlots);
    // Publication is an external consumer: a single-output fusion must not
    // eliminate a value that the caller requested alongside downstream results.
    if (requestedOutputSlots != nullptr) {
        for (int i = 0; i < numRequestedOutputs; ++i) {
            if (requestedOutputSlots[i] >= 0) ++consumerCounts[requestedOutputSlots[i]];
        }
    }

    // Track which slots are already part of a fusion (no overlapping fusions)
    std::vector<bool> fused(numSlots, false);

    // Flat output indices are not producer slot indices. Build dtype evidence
    // once, including multi-output producers, and update it when casts become
    // identities so later decisions cannot use their obsolete target dtype.
    std::unordered_map<int, DataType> outputTypes;
    auto sourceType = [&outputTypes](int source) {
        auto it = outputTypes.find(source);
        return source >= 0 && it != outputTypes.end() ? it->second : DataType::UNKNOWN;
    };
    for (int s = 0; s < numSlots; ++s) {
        for (int o = 0; o < slots[s].wiring.numOutputs; ++o) {
            outputTypes[slots[s].wiring.outputSlotIndices[o]] =
                slots[s].isIdentityOp() && slots[s].wiring.numInputs == 1
                ? sourceType(slots[s].wiring.inputSourceIndices[0])
                : retainedOutputType(slots[s], o);
        }
    }

    // Pass 0: Only A→B→A with a proven lossless widening A→B is redundant.
    // Narrowing round trips carry observable rounding/overflow semantics.
    if (Environment::getInstance().dspCastElimination()) {
        int castsEliminated = 0;
        for (int i = 0; i < numSlots; i++) {
            if (fused[i]) continue;
            if (!slotHasTrait(slots[i], sd::ops::OP_TRAIT_CAST) || slots[i].isIdentityOp()) continue;
            if (slots[i].wiring.numOutputs != 1 || slots[i].wiring.numInputs != 1) continue;
            if (slots[i].args.numIArgs < 1) continue;

            const auto inputType = sourceType(slots[i].wiring.inputSourceIndices[0]);
            const auto intermediateType = static_cast<DataType>(slots[i].args.iArgs[0]);
            if (!isLosslessFloatWidening(inputType, intermediateType)) continue;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, i)) continue;
            int outputIdx = slots[i].wiring.outputSlotIndices[0];

            // Find the consumer of this cast's output
            for (int j = i + 1; j < numSlots; j++) {
                if (fused[j]) continue;
                if (!slotHasTrait(slots[j], sd::ops::OP_TRAIT_CAST) || slots[j].isIdentityOp()) continue;
                if (slots[j].wiring.numInputs != 1 || slots[j].wiring.numOutputs != 1) continue;
                if (slots[j].args.numIArgs < 1) continue;
                if (slots[j].wiring.inputSourceIndices[0] != outputIdx) continue;

                if (static_cast<DataType>(slots[j].args.iArgs[0]) != inputType) continue;

                // Mark both as identity ops (skip execution, wire through)
                slots[i].addOpTrait(sd::ops::OP_TRAIT_IDENTITY);
                slots[j].addOpTrait(sd::ops::OP_TRAIT_IDENTITY);
                fused[i] = true;
                fused[j] = true;
                outputTypes[outputIdx] = inputType;
                outputTypes[slots[j].wiring.outputSlotIndices[0]] = inputType;
                castsEliminated += 2;
                break;
            }
        }
        if (castsEliminated > 0) {
            DSP_DIAG(FUSION, "cast elimination removed %d redundant cast ops", castsEliminated);
        }
    }

    // Pass 0.5: Sink proven HALF→FLOAT32 casts only into dense mixed-input
    // matmuls. Every consumer must preserve the FLOAT32 output contract and
    // the cast result must not itself be a requested output.
    if (Environment::getInstance().dspCastSinkMatmul()) {
        int castsSunk = 0;
        int castsSkippedNonMatmulConsumer = 0;
        int totalCasts = 0;
        int castsFp32Target = 0;
        int castsNoIArgs = 0;
        int castsMultiIO = 0;
        int castsNoConsumer = 0;
        for (int i = 0; i < numSlots; i++) {
            if (fused[i]) continue;
            if (!slotHasTrait(slots[i], sd::ops::OP_TRAIT_CAST) || slots[i].isIdentityOp()) continue;
            totalCasts++;
            if (slots[i].wiring.numOutputs != 1 || slots[i].wiring.numInputs != 1) { castsMultiIO++; continue; }
            if (slots[i].args.numIArgs < 1) { castsNoIArgs++; continue; }

            int targetType = static_cast<int>(slots[i].args.iArgs[0]);
            if (targetType != static_cast<int>(FLOAT32)) continue;
            castsFp32Target++;
            // Dense mixed HALF/FLOAT32 is supported. DOUBLE→FLOAT32 is
            // narrowing; BF16/integer/unknown inputs have no proven dense ABI.
            if (sourceType(slots[i].wiring.inputSourceIndices[0]) != HALF) continue;

            int outputIdx = slots[i].wiring.outputSlotIndices[0];
            if (requestedOutputSlots != nullptr &&
                std::find(requestedOutputSlots, requestedOutputSlots + numRequestedOutputs,
                          outputIdx) != requestedOutputSlots + numRequestedOutputs) continue;

            // Check ALL consumers of this cast's output — every one must be a matmul
            bool allConsumersAreMatmul = true;
            int consumerCount = 0;
            for (int j = i + 1; j < numSlots; j++) {
                if (fused[j]) continue;
                bool consumesOutput = false;
                for (int k = 0; k < slots[j].wiring.numInputs; k++) {
                    if (slots[j].wiring.inputSourceIndices[k] == outputIdx) {
                        consumesOutput = true;
                        break;
                    }
                }
                if (!consumesOutput) continue;

                consumerCount++;
                if (!isDenseMmulSlot(slots[j])) {
                    allConsumersAreMatmul = false;
                    break;
                }
                // Keep the other operand FLOAT32: sinking both operands would
                // change matmul's inferred output dtype and rounding to HALF.
                for (int k = 0; k < slots[j].wiring.numInputs; ++k) {
                    if (slots[j].wiring.inputSourceIndices[k] == outputIdx) {
                        if (sourceType(slots[j].wiring.inputSourceIndices[1 - k]) != FLOAT32 ||
                            slots[j].wiring.inputSourceIndices[1 - k] == outputIdx) {
                            allConsumersAreMatmul = false;
                        }
                    }
                }
                if (!allConsumersAreMatmul) break;
            }

            if (consumerCount > 0 && allConsumersAreMatmul) {
                // MmulHelper owns the supported HALF/FLOAT32 conversion.
                slots[i].addOpTrait(sd::ops::OP_TRAIT_IDENTITY);
                fused[i] = true;
                outputTypes[outputIdx] = HALF;
                castsSunk++;
            } else if (consumerCount > 0) {
                castsSkippedNonMatmulConsumer++;
            } else {
                castsNoConsumer++;
            }
        }
        DSP_DIAG(FUSION, "cast sink: total=%d fp32Target=%d sunk=%d skippedNonMatmul=%d noConsumer=%d",
                 totalCasts, castsFp32Target, castsSunk, castsSkippedNonMatmulConsumer, castsNoConsumer);
        if (castsSunk > 0 || castsSkippedNonMatmulConsumer > 0) {
            DSP_DIAG(FUSION, "cast sink through matmul: %d sunk, %d skipped (non-matmul consumer)",
                     castsSunk, castsSkippedNonMatmulConsumer);
        }
    }

    // Pass 1: Detect element-wise chains
    DSP_DIAG(FUSION, "detectFusions: pass 1 — element-wise chain detection");
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isElementwiseSlot(slots[i])) continue;
        if (slots[i].wiring.numOutputs != 1) continue;  // Only single-output ops

        // Try to extend a chain starting at slot i
        std::vector<int> chain;
        chain.push_back(i);
        int current = i;

        while (chain.size() < static_cast<size_t>(MAX_CHAIN_LENGTH)) {
            int outputIdx = slots[current].wiring.outputSlotIndices[0];

            // Find the next slot that consumes this output
            int nextSlot = -1;
            for (int j = current + 1; j < numSlots; j++) {
                if (fused[j]) continue;
                for (int k = 0; k < slots[j].wiring.numInputs; k++) {
                    if (slots[j].wiring.inputSourceIndices[k] == outputIdx) {
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
            if (slots[nextSlot].wiring.numOutputs != 1) break;

            chain.push_back(nextSlot);
            current = nextSlot;
        }

        // Only create a fusion candidate if chain has >= 2 ops
        if (chain.size() >= 2) {
            DSP_DIAG(FUSION, "pass 1: element-wise chain [%d-%d] length=%d head=%s",
                      chain.front(), chain.back(), (int)chain.size(),
                      slots[chain.front()].ident.opName.c_str());
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

    // Pass 2: Reduction epilogue fusion (complements Pass 4 with isOnlyConsumedOnce + isFusedChainTail)
    fuseReductionEpilogues(slots, numSlots, fused, consumerCounts);

    // Pass 3: Normalization epilogue fusion (complements Pass 5 with isOnlyConsumedOnce + isFusedChainTail)
    fuseNormalizationEpilogues(slots, numSlots, fused, consumerCounts);

    // Pass 4: Detect bias+activation patterns (add -> relu/sigmoid/tanh/gelu)
    DSP_DIAG(FUSION, "detectFusions: pass 4 — bias+activation pattern detection");
    for (int i = 0; i < numSlots - 1; i++) {
        if (fused[i]) continue;

        std::string opName = getOpName(slots[i]);
        if (opName != "add") continue;
        if (slots[i].wiring.numOutputs != 1) continue;

        int outputIdx = slots[i].wiring.outputSlotIndices[0];

        // Find activation consuming this add's output
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            if (!slotHasTrait(slots[j], sd::ops::OP_TRAIT_ACTIVATION)) continue;
            if (slots[j].wiring.numInputs != 1) continue;
            if (slots[j].wiring.inputSourceIndices[0] != outputIdx) continue;
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

    // Pass 5: Detect matmul → add(bias) → optional activation patterns
    DSP_DIAG(FUSION, "detectFusions: pass 5 — matmul+bias+activation pattern detection");
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;

        if (!isDenseMmulSlot(slots[i])) continue;
        if (slots[i].wiring.numOutputs != 1) continue;

        int matmulOutputIdx = slots[i].wiring.outputSlotIndices[0];

        // Find add consuming matmul's output
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string addName = getOpName(slots[j]);
            if (addName != "add") continue;
            if (slots[j].wiring.numInputs != 2) continue;

            // One input must be matmul output, other must be a 1D bias vector.
            // 2D operands are residual adds (not bias) and must not be fused.
            bool foundMatmulOutput = false;
            int externalSrcIdx = INT_MIN;  // raw inputSourceIndices value (negative)
            for (int k = 0; k < slots[j].wiring.numInputs; k++) {
                if (slots[j].wiring.inputSourceIndices[k] == matmulOutputIdx) {
                    foundMatmulOutput = true;
                } else if (slots[j].wiring.inputSourceIndices[k] < 0) {
                    externalSrcIdx = slots[j].wiring.inputSourceIndices[k];
                }
            }
            if (!foundMatmulOutput || externalSrcIdx == INT_MIN) continue;
            if (!isOnlyConsumedOnce(consumerCounts, slots, numSlots, i)) continue;
            if (slots[j].wiring.numOutputs != 1) continue;

            // Convert source index to external input array index: srcIdx = -(extIdx+1)
            int extIdx = -(externalSrcIdx + 1);
            if (extIdx < 0 || extIdx >= static_cast<int>(externalInputRanks.size())) continue;
            int extRank = externalInputRanks[extIdx];
            if (extRank != 1) continue;  // residual or higher-dim — not a bias, skip fusion

            // Found matmul → add. Now check for optional activation.
            std::vector<int> chain = {i, j};
            int addOutputIdx = slots[j].wiring.outputSlotIndices[0];

            for (int a = j + 1; a < numSlots; a++) {
                if (fused[a]) continue;
                if (!slotHasTrait(slots[a], sd::ops::OP_TRAIT_ACTIVATION)) continue;
                if (slots[a].wiring.numInputs != 1) continue;
                if (slots[a].wiring.inputSourceIndices[0] != addOutputIdx) continue;
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

    // Pass 6: Detect reduction → element-wise chains (e.g., reduce_sum → sqrt for norm)
    DSP_DIAG(FUSION, "detectFusions: pass 6 — reduction+element-wise chain detection");
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isReductionSlot(slots[i])) continue;
        if (slots[i].wiring.numOutputs != 1) continue;

        auto chain = extendElementwiseChain(i, slots, numSlots, fused, consumerCounts);
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

    // Pass 7: Detect normalization → element-wise chains (e.g., softmax → log)
    DSP_DIAG(FUSION, "detectFusions: pass 7 — normalization+element-wise chain detection");
    for (int i = 0; i < numSlots; i++) {
        if (fused[i]) continue;
        if (!isNormalizationSlot(slots[i])) continue;
        if (slots[i].wiring.numOutputs != 1) continue;

        auto chain = extendElementwiseChain(i, slots, numSlots, fused, consumerCounts);
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

    // Pass 8: Detect softmax compound patterns (reduce_max → sub → exp → reduce_sum → div)
    DSP_DIAG(FUSION, "detectFusions: pass 8 — softmax compound pattern detection");
    for (int i = 0; i < numSlots - 4; i++) {
        if (fused[i]) continue;
        std::string op0 = getOpName(slots[i]);
        if (op0 != "reduce_max") continue;
        if (slots[i].wiring.numOutputs != 1) continue;

        int out0 = slots[i].wiring.outputSlotIndices[0];
        // Find sub consuming reduce_max output
        int subSlot = -1;
        for (int j = i + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "subtract") continue;
            for (int k = 0; k < slots[j].wiring.numInputs; k++) {
                if (slots[j].wiring.inputSourceIndices[k] == out0) { subSlot = j; break; }
            }
            if (subSlot >= 0) break;
        }
        if (subSlot < 0) continue;

        int outSub = slots[subSlot].wiring.outputSlotIndices[0];
        // Find exp consuming sub output
        int expSlot = -1;
        for (int j = subSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "exp") continue;
            if (slots[j].wiring.numInputs >= 1 && slots[j].wiring.inputSourceIndices[0] == outSub) {
                expSlot = j; break;
            }
        }
        if (expSlot < 0) continue;

        int outExp = slots[expSlot].wiring.outputSlotIndices[0];
        // Find reduce_sum consuming exp output
        int sumSlot = -1;
        for (int j = expSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "reduce_sum") continue;
            if (slots[j].wiring.numInputs >= 1 && slots[j].wiring.inputSourceIndices[0] == outExp) {
                sumSlot = j; break;
            }
        }
        if (sumSlot < 0) continue;

        int outSum = slots[sumSlot].wiring.outputSlotIndices[0];
        // Find div consuming exp and sum outputs
        int divSlot = -1;
        for (int j = sumSlot + 1; j < numSlots; j++) {
            if (fused[j]) continue;
            std::string opJ = getOpName(slots[j]);
            if (opJ != "divide") continue;
            bool hasExp = false, hasSum = false;
            for (int k = 0; k < slots[j].wiring.numInputs; k++) {
                if (slots[j].wiring.inputSourceIndices[k] == outExp) hasExp = true;
                if (slots[j].wiring.inputSourceIndices[k] == outSum) hasSum = true;
            }
            if (hasExp && hasSum) { divSlot = j; break; }
        }
        if (divSlot < 0) continue;

        // Only the pattern's own consumers may read the intermediates: exp's
        // output feeds reduce_sum and divide, the others feed one member each.
        auto consumersOf = [&consumerCounts](int output) {
            auto it = consumerCounts.find(output);
            return it == consumerCounts.end() ? 0 : it->second;
        };
        if (consumersOf(outSub) != 1 || consumersOf(outExp) != 2 || consumersOf(outSum) != 1) continue;

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

        DSP_DIAG(FUSION, "detected softmax compound pattern, slots %d-%d",
                  chain.front(), chain.back());
    }

    // Sort by startSlot for deterministic ordering
    std::sort(candidates.begin(), candidates.end(),
              [](const FusionCandidate& a, const FusionCandidate& b) {
                  return a.startSlot < b.startSlot;
              });

    // Summary: count by type
    int numElemChain = 0, numBiasAct = 0, numMatmulBias = 0;
    for (const auto& c : candidates) {
        switch (c.type) {
            case FusionCandidate::ELEMENTWISE_CHAIN: numElemChain++; break;
            case FusionCandidate::BIAS_ACTIVATION: numBiasAct++; break;
            case FusionCandidate::MATMUL_BIAS_ACTIVATION: numMatmulBias++; break;
        }
    }
    DSP_DIAG(FUSION, "detectFusions: END %d candidates (elemChain=%d biasAct=%d matmulBias=%d)",
             (int)candidates.size(), numElemChain, numBiasAct, numMatmulBias);

    return candidates;
}

#if NOT_EXCLUDED(OP_fused_elementwise_chain)
static_assert(MAX_FUSED_CHAIN == sd::ops::helpers::FUSED_CHAIN_MAX_OPS,
              "FusedChain metadata and the fused kernel must agree on the chain length");

// A fused member must compute exactly what its slot computes eagerly: the eager
// op's functor with the eager parameters, the chain value as the functor's first
// operand, rounded to the storage type. The tables below list only ops whose
// eager kernel is that functor; every other op keeps its own slot.
struct FusedMember {
    int code = -1;
    int secondarySource = FusionPass::FUSED_NO_SECONDARY_SOURCE;
    int8_t policy = FUSED_SECONDARY_NONE;
    bool hasClip = false;
    double clipMin = 0.0;
    double clipMax = 0.0;
};

// Numbering from loops/legacy_ops.h TRANSFORM_{SAME,STRICT,FLOAT}_OPS.
static int legacyTransformCode(int legacyOpType, int opNum) {
    using namespace sd::ops::helpers;
    switch (legacyOpType) {
        case LEGACY_TRANSFORM_SAME:
            switch (opNum) {
                case 0: return FUSED_ABS;
                case 1: return FUSED_SIGN;
                case 3: return FUSED_NEG;
                case 4: return FUSED_ROUND;
                case 11: return FUSED_RECIPROCAL;
                case 12: return FUSED_SQUARE;
                case 17: return FUSED_CEIL;
                case 18: return FUSED_FLOOR;
                default: return -1;
            }
        case LEGACY_TRANSFORM_STRICT:
            switch (opNum) {
                case 22: return FUSED_COS;
                case 23: return FUSED_EXP;
                case 24: return FUSED_LOG;
                case 26: return FUSED_SIGMOID;
                case 27: return FUSED_SIN;
                case 28: return FUSED_SOFTPLUS;
                case 29: return FUSED_TANH;
                case 33: return FUSED_HARDTANH;
                case 34: return FUSED_SOFTSIGN;
                case 36: return FUSED_HARD_SIGMOID;
                case 42: return FUSED_SELU;
                case 43: return FUSED_SWISH;
                case 44: return FUSED_LOG1P;
                case 45: return FUSED_ERF;
                case 50: return FUSED_ERFC;
                case 53: return FUSED_GELU;
                case 57: return FUSED_MISH;
                default: return -1;
            }
        case LEGACY_TRANSFORM_FLOAT:
            switch (opNum) {
                case 1: return FUSED_SQRT;
                case 3: return FUSED_RSQRT;
                default: return -1;
            }
        default:
            return -1;
    }
}

// SCALAR_OPS with the scalar supplied as input 1 (element 0 is read).
static int legacyScalarBinaryCode(int opNum) {
    using namespace sd::ops::helpers;
    switch (opNum) {
        case 0: return FUSED_ADD;
        case 1: return FUSED_SUB;
        case 2: return FUSED_MUL;
        case 3: return FUSED_DIV;
        case 4: return FUSED_REVERSE_DIV;
        case 5: return FUSED_REVERSE_SUB;
        case 6: return FUSED_MAX;
        case 13: return FUSED_MIN;
        case 15: return FUSED_MOD;
        case 20: return FUSED_FLOORDIV;
        case 22: return FUSED_SQUARED_SUB;
        case 26: return FUSED_ATAN2;
        case 31: return FUSED_POW;
        case 35: return FUSED_LEAKY_RELU;
        default: return -1;
    }
}

// SCALAR_OPS with the scalar in tArgs[0]. The fused kernel has no scalar
// parameter, so only the constants its unary functors hard-code are accepted.
static int legacyScalarUnaryCode(int opNum, const NativeSlot& slot) {
    using namespace sd::ops::helpers;
    if (slot.args.numTArgs != 1 || slot.args.tArgs == nullptr) return -1;
    const double scalar = slot.args.tArgs[0];
    switch (opNum) {
        case 39: return scalar == 0.0 && !std::signbit(scalar) ? FUSED_RELU : -1;
        case 40: return scalar == 0.0 && !std::signbit(scalar) ? FUSED_RELU6 : -1;
        case 7: return scalar == 1.0 ? FUSED_ELU : -1;
        default: return -1;
    }
}

// PAIRWISE_TRANSFORM_OPS numbering.
static int legacyPairwiseCode(int opNum) {
    using namespace sd::ops::helpers;
    switch (opNum) {
        case 0: return FUSED_ADD;
        case 2: return FUSED_DIV;
        case 3: return FUSED_MUL;
        case 4: return FUSED_POW;
        case 5: return FUSED_REVERSE_SUB;
        case 6: return FUSED_SUB;
        case 7: return FUSED_MAX;
        case 8: return FUSED_MIN;
        case 11: return FUSED_REVERSE_DIV;
        case 16: return FUSED_ATAN2;
        case 18: return FUSED_FLOORDIV;
        case 20: return FUSED_SQUARED_SUB;
        case 23: return FUSED_MOD;
        default: return -1;
    }
}

// Broadcastable declarables whose eager kernel is the functor of the code.
static int declarableBinaryCode(const std::string& name) {
    using namespace sd::ops::helpers;
    if (name == "add") return FUSED_ADD;
    if (name == "subtract") return FUSED_SUB;
    if (name == "multiply") return FUSED_MUL;
    if (name == "divide" || name == "realdiv") return FUSED_DIV;
    if (name == "reversesubtract") return FUSED_REVERSE_SUB;
    if (name == "reversedivide") return FUSED_REVERSE_DIV;
    if (name == "squaredsubtract") return FUSED_SQUARED_SUB;
    if (name == "maximum") return FUSED_MAX;
    if (name == "minimum") return FUSED_MIN;
    if (name == "mod") return FUSED_MOD;
    if (name == "pow") return FUSED_POW;
    return -1;
}

// Unary declarables; parameterised ones only with the parameter the fused
// functor applies.
static int declarableUnaryCode(const std::string& name, const NativeSlot& slot, FusedMember& member) {
    using namespace sd::ops::helpers;
    const int numT = slot.args.tArgs == nullptr ? 0 : slot.args.numTArgs;
    const double* t = slot.args.tArgs;
    if (name == "sigmoid") return FUSED_SIGMOID;
    if (name == "tanh") return FUSED_TANH;
    if (name == "softsign") return FUSED_SOFTSIGN;
    if (name == "softplus") return FUSED_SOFTPLUS;
    if (name == "selu") return FUSED_SELU;
    if (name == "hardtanh") return FUSED_HARDTANH;
    if (name == "hardsigmoid") return FUSED_HARD_SIGMOID;
    if (name == "silu") return FUSED_SILU;
    if (name == "relu" || name == "relu6") {
        if (numT > 0 && (t[0] != 0.0 || std::signbit(t[0]))) return -1;
        return name == "relu" ? FUSED_RELU : FUSED_RELU6;
    }
    if (name == "elu") return numT == 0 || t[0] == 1.0 ? FUSED_ELU : -1;
    if (name == "clipbyvalue") {
        if (numT < 2 || !(t[0] < t[1])) return -1;
        member.hasClip = true;
        member.clipMin = t[0];
        member.clipMax = t[1];
        return FUSED_CLIP;
    }
    return -1;
}

/**
 * Resolves one chain member. chainInput is the input that carries the chain
 * value, or -1 for a run head (whose chain value is input 0).
 */
static bool resolveFusedMember(const NativeSlot& slot, int chainInput, FusedMember& member) {
    using namespace sd::ops::helpers;
    member = FusedMember{};
    if (slot.wiring.numOutputs != 1 || slot.hasValueDependentShape() || !isElementwiseSlot(slot) ||
        slot.aliasesInput()) {
        return false;
    }
    const int numInputs = slot.wiring.numInputs;
    if (numInputs < 1 || numInputs > 2 || slot.wiring.inputSourceIndices == nullptr) return false;

    int code = -1;
    int8_t policy = FUSED_SECONDARY_NONE;
    const int legacyType = slot.legacy.legacyOpType;
    if (legacyType != LEGACY_NOT_SET) {
        const int opNum = slot.legacy.legacyOpNum;
        switch (legacyType) {
            case LEGACY_TRANSFORM_SAME:
            case LEGACY_TRANSFORM_STRICT:
            case LEGACY_TRANSFORM_FLOAT:
                if (numInputs == 1) code = legacyTransformCode(legacyType, opNum);
                break;
            case LEGACY_SCALAR:
                if (numInputs == 2) {
                    code = legacyScalarBinaryCode(opNum);
                    policy = FUSED_SECONDARY_SCALAR;
                } else {
                    code = legacyScalarUnaryCode(opNum, slot);
                }
                break;
            case LEGACY_PAIRWISE_TRANSFORM:
                if (numInputs == 2) {
                    code = legacyPairwiseCode(opNum);
                    policy = FUSED_SECONDARY_SAME_SHAPE;
                }
                break;
            default:
                break;
        }
    } else if (slot.ident.op != nullptr && slot.ident.op->getOpName() != nullptr) {
        std::string name = *slot.ident.op->getOpName();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (numInputs == 2) {
            code = declarableBinaryCode(name);
            policy = FUSED_SECONDARY_BROADCAST;
        } else {
            code = declarableUnaryCode(name, slot, member);
        }
    }
    if (code < 0 || !isImplementedFusedOp(code)) return false;
    const bool binary = isBinaryFusedOp(static_cast<FusedElemOp>(code));
    if (binary != (numInputs == 2)) return false;

    if (binary) {
        const int src0 = slot.wiring.inputSourceIndices[0];
        const int src1 = slot.wiring.inputSourceIndices[1];
        if (chainInput <= 0) {
            // A non-head member cannot name the chain value as its secondary:
            // the intermediate is never materialised.
            if (chainInput == 0 && src0 == src1) return false;
            member.secondarySource = src1;
        } else if (chainInput == 1) {
            // A legacy scalar op reads element 0 of input 1; the chain value
            // cannot take that role.
            if (src0 == src1 || policy == FUSED_SECONDARY_SCALAR) return false;
            code = swappedBinaryFusedCode(code);
            if (code < 0) return false;
            member.secondarySource = src0;
        } else {
            return false;
        }
        member.policy = policy;
    } else if (chainInput > 0) {
        return false;
    }
    member.code = code;
    return true;
}

static void commitFusedRun(NativeSlot* slots, const std::vector<int>& run,
                           const std::vector<FusedMember>& members) {
    FusedChain& chain = slots[run.front()].fusedChain;
    chain.clearHead();
    chain.isFusedChainHead = true;
    chain.fusedChainLength = static_cast<int>(run.size());
    for (size_t i = 0; i < run.size(); i++) {
        const FusedMember& member = members[i];
        chain.fusedChainOpCodes[i] = member.code;
        chain.fusedChainSlots[i] = run[i];
        chain.fusedChainSecondaryInputSources[i] = member.secondarySource;
        chain.fusedChainSecondaryPolicy[i] = member.policy;
        if (member.hasClip) {
            chain.fusedChainHasClip = true;
            chain.fusedChainClipMin = member.clipMin;
            chain.fusedChainClipMax = member.clipMax;
        }
    }
    for (size_t i = 1; i < run.size(); i++) slots[run[i]].fusedChain.isFusedChainTail = true;
}

/**
 * Splits a candidate chain into fused runs of 2..MAX_FUSED_CHAIN members. A
 * member extends the current run only when it consumes the previous member's
 * sole-consumer output, resolves at that input, keeps the chain's single clip
 * pair and reads a secondary that already exists when the head executes: the
 * whole run executes at the head's position, so an internal secondary must be
 * produced before the head. Anything else ends the run and may start a new one.
 */
static int buildFusedRuns(NativeSlot* slots, int numSlots, const std::vector<int>& chain,
                          const std::unordered_map<int, int>& consumerCounts,
                          const std::unordered_map<int, int>& producerOf) {
    int committed = 0;
    std::vector<int> run;
    std::vector<FusedMember> members;
    auto flush = [&]() {
        if (run.size() >= 2) {
            commitFusedRun(slots, run, members);
            committed++;
            DSP_DIAG(FUSION, "fused kernel dispatch enabled for chain slots %d-%d (%d ops)",
                     run.front(), run.back(), (int)run.size());
        }
        run.clear();
        members.clear();
    };

    for (int slotIdx : chain) {
        if (slotIdx < 0 || slotIdx >= numSlots) {
            flush();
            continue;
        }
        const NativeSlot& slot = slots[slotIdx];
        FusedMember member;
        bool extended = false;
        if (!run.empty() && run.size() < static_cast<size_t>(MAX_FUSED_CHAIN)) {
            const int prevOutput = slots[run.back()].wiring.outputSlotIndices[0];
            auto count = consumerCounts.find(prevOutput);
            int chainInput = -1;
            for (int k = 0; k < slot.wiring.numInputs; k++) {
                if (slot.wiring.inputSourceIndices[k] == prevOutput) {
                    chainInput = k;
                    break;
                }
            }
            if (count != consumerCounts.end() && count->second == 1 && chainInput >= 0 &&
                resolveFusedMember(slot, chainInput, member)) {
                bool compatible = true;
                if (member.hasClip) {
                    for (const FusedMember& earlier : members) {
                        if (earlier.hasClip &&
                            (earlier.clipMin != member.clipMin || earlier.clipMax != member.clipMax)) {
                            compatible = false;
                        }
                    }
                }
                if (compatible && member.secondarySource >= 0) {
                    auto producer = producerOf.find(member.secondarySource);
                    compatible = producer != producerOf.end() && producer->second < run.front();
                }
                if (compatible) {
                    run.push_back(slotIdx);
                    members.push_back(member);
                    extended = true;
                }
            }
        }
        if (!extended) {
            flush();
            if (resolveFusedMember(slot, -1, member)) {
                run.push_back(slotIdx);
                members.push_back(member);
            }
        }
    }
    flush();
    return committed;
}
#endif

int FusionPass::applyFusions(
        NativeSlot* slots, int numSlots,
        const std::vector<FusionCandidate>& candidates,
        const int* requestedOutputSlots, int numRequestedOutputs) {

    int applied = 0;

    DSP_DIAG(FUSION, "applyFusions: BEGIN numSlots=%d candidates=%d", numSlots, (int)candidates.size());
    if (slots == nullptr || numSlots <= 0) return 0;

    // Fused-chain metadata is derived only here. A re-freeze must not inherit a
    // chain an earlier freeze built, even when no candidate survives this time.
    for (int s = 0; s < numSlots; s++) {
        slots[s].fusedChain.clearHead();
        slots[s].fusedChain.isFusedChainTail = false;
    }

    // Requested outputs are consumers: a chain must neither overwrite nor skip
    // producing a value the caller reads.
    auto consumerCounts = buildConsumerCounts(slots, numSlots);
    if (requestedOutputSlots != nullptr) {
        for (int i = 0; i < numRequestedOutputs; ++i) {
            if (requestedOutputSlots[i] >= 0) ++consumerCounts[requestedOutputSlots[i]];
        }
    }
    std::unordered_map<int, int> producerOf;
    for (int s = 0; s < numSlots; s++) {
        for (int o = 0; o < slots[s].wiring.numOutputs; o++) producerOf[slots[s].wiring.outputSlotIndices[o]] = s;
    }

    // Member i may overwrite member i-1's output only when it is that output's
    // sole consumer, computes elementwise (reads each element before writing
    // it), and the output is a buffer the producer owns rather than an alias of
    // its input or a frozen constant.
    auto enableChainInPlace = [&](const std::vector<int>& chain) {
        for (size_t i = 1; i < chain.size(); i++) {
            const int slotIdx = chain[i];
            const int prevSlotIdx = chain[i - 1];
            if (slotIdx < 0 || slotIdx >= numSlots || prevSlotIdx < 0 || prevSlotIdx >= numSlots) continue;
            NativeSlot& slot = slots[slotIdx];
            const NativeSlot& prev = slots[prevSlotIdx];
            if (!isElementwiseSlot(slot) || prev.wiring.numOutputs < 1 || prev.aliasesInput() ||
                prev.frozenConstantSlot()) {
                continue;
            }
            const int prevOutputSlot = prev.wiring.outputSlotIndices[0];
            auto count = consumerCounts.find(prevOutputSlot);
            if (count == consumerCounts.end() || count->second != 1) continue;
            for (int k = 0; k < slot.wiring.numInputs; k++) {
                if (slot.wiring.inputSourceIndices[k] == prevOutputSlot) {
                    slot.enableInPlaceFusion(k);
                    break;
                }
            }
        }
    };

    for (const auto& fusion : candidates) {
        // Multi-GPU: never fuse a chain whose slots span a device boundary. assignDevices()
        // partitions ops across GPUs by targetDeviceId, and that partition can fall in the
        // middle of a fusible elementwise run (e.g. add on dev0, tanh on dev1). Fusing across
        // the split makes the cross-device op a fused-chain TAIL, which the shape pre-pass and
        // slot executor skip for independent output allocation — leaving an uninitialized
        // NDArray that crashes at freeze/resegment (buildSegments -> estimateSlotOutputBytes).
        // Leave such ops unfused; they run as separate slots and platformMigrateSegmentInputs
        // handles the cross-device data transfer between them.
        {
            bool crossDevice = false, firstDev = true;
            int chainDev = -1;
            for (int idx : fusion.slotIndices) {
                if (idx < 0 || idx >= numSlots) continue;
                int d = slots[idx].targetDeviceId;
                if (firstDev) { chainDev = d; firstDev = false; }
                else if (d != chainDev) { crossDevice = true; break; }
            }
            if (crossDevice) {
                DSP_DIAG(FUSION, "applyFusions: SKIP cross-device fusion candidate "
                         "(slots span multiple targetDeviceId)");
                continue;
            }
        }
        switch (fusion.type) {

            case FusionCandidate::ELEMENTWISE_CHAIN: {
                if (fusion.slotIndices.size() < 2) break;
                enableChainInPlace(fusion.slotIndices);
#if NOT_EXCLUDED(OP_fused_elementwise_chain)
                buildFusedRuns(slots, numSlots, fusion.slotIndices, consumerCounts, producerOf);
#endif
                applied++;
                DSP_DIAG(FUSION, "applied ELEMENTWISE_CHAIN fusion, slots %d-%d (%d ops)",
                          fusion.startSlot, fusion.endSlot, fusion.chainLength);
                break;
            }

            case FusionCandidate::BIAS_ACTIVATION: {
                // add(x, bias) → activation(result): the activation reuses the add's buffer.
                if (fusion.slotIndices.size() != 2) break;
                const int actSlotIdx = fusion.slotIndices[1];
                if (actSlotIdx < 0 || actSlotIdx >= numSlots) break;
                enableChainInPlace(fusion.slotIndices);
                if (slots[actSlotIdx].isInPlaceFused()) {
                    applied++;
                    DSP_DIAG(FUSION, "applied BIAS_ACTIVATION fusion, slots %d-%d",
                              fusion.startSlot, fusion.endSlot);
                }
                break;
            }

            case FusionCandidate::MATMUL_BIAS_ACTIVATION: {
                // matmul → add(bias) → optional activation. The add and the activation
                // still execute, so they only reuse the matmul's buffer. A cuBLASLt bias
                // epilogue on the matmul would apply the bias a second time.
                if (fusion.slotIndices.size() < 2) break;
                const int matmulSlotIdx = fusion.slotIndices[0];
                if (matmulSlotIdx < 0 || matmulSlotIdx >= numSlots) break;
                if (!isDenseMmulSlot(slots[matmulSlotIdx])) break;
                enableChainInPlace(fusion.slotIndices);
                applied++;
                DSP_DIAG(FUSION, "applied MATMUL_BIAS_ACTIVATION in-place fusion, slots %d-%d",
                          fusion.startSlot, fusion.endSlot);
                break;
            }
        }
    }

    DSP_DIAG(FUSION, "applyFusions: END applied=%d of %d candidates", applied, (int)candidates.size());

    return applied;
}

}  // namespace graph
}  // namespace sd
