/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sampling_penalties)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/sampling_penalties.h>

namespace sd {
namespace ops {

/**
 * Apply repetition/frequency/presence penalties and min-P filtering to logits.
 *
 * Inputs:
 *   0: logits   [batch, seqLen, vocabSize], [batch, vocabSize], or [vocabSize] — modified in output
 *   1: inputIds [batch, seqLen], [seqLen], or scalar INT64 — prior tokens for penalty computation
 *
 * Output:
 *   0: penalized logits (same shape and type as input 0)
 *
 * Float args:
 *   0: repetitionPenalty (1.0 = off)
 *   1: frequencyPenalty (0.0 = off)
 *   2: presencePenalty (0.0 = off)
 *   3: minP (0.0 = off)
 */
CUSTOM_OP_IMPL(sampling_penalties, 2, 1, false, 0, 0) {
    auto logits = INPUT_VARIABLE(0);
    auto inputIds = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    auto logitsRank = logits->rankOf();
    REQUIRE_TRUE(logitsRank >= 1 && logitsRank <= 3, 0,
                 "sampling_penalties: logits must be rank 1, 2, or 3, got %lld",
                 (long long)logitsRank);

    auto idsRank = inputIds->rankOf();
    REQUIRE_TRUE(idsRank == 0 || idsRank == 1 || idsRank == 2, 0,
                 "sampling_penalties: inputIds must be scalar, rank 1, or rank 2, got %lld",
                 (long long)idsRank);

    double repPenalty = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
    double freqPenalty = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0;
    double presPenalty = block.getTArguments()->size() > 2 ? T_ARG(2) : 0.0;
    double minP = block.getTArguments()->size() > 3 ? T_ARG(3) : 0.0;

    // Copy logits to output if not in-place
    if (logits != output) {
        output->assign(logits);
    }

    // Apply penalties
    if (repPenalty != 1.0 || freqPenalty != 0.0 || presPenalty != 0.0) {
        helpers::applyLogitPenalties(output, inputIds, repPenalty, freqPenalty, presPenalty,
                                     block.launchContext());
    }

    // Apply min-P filtering
    if (minP > 0.0) {
        helpers::applyMinPFilter(output, minP, block.launchContext());
    }

    return sd::Status::OK;
}

DECLARE_TYPES(sampling_penalties) {
  getOpDescriptor()->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(1, {INT64, INT32});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(sampling_penalties) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

/**
 * typical_p_filter - Typical-p (entropy-deviation) logit filter. Masks tokens whose
 * information content -log(p) deviates most from the distribution entropy, keeping the
 * tokens with smallest deviation until their cumulative mass >= typicalP. Modifies logits
 * in place (masked positions set to -inf). typicalP in (0,1) enables; 1.0 = no-op.
 * Float args: 0: typicalP (1.0 = off)
 */
CUSTOM_OP_IMPL(typical_p_filter, 1, 1, false, 1, 0) {
    auto logits = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    auto logitsRank = logits->rankOf();
    REQUIRE_TRUE(logitsRank == 1 || logitsRank == 2, 0,
                 "typical_p_filter: logits must be rank 1 or 2, got %lld", (long long)logitsRank);

    double typicalP = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    if (logits != output) output->assign(logits);

    if (typicalP > 0.0 && typicalP < 1.0) {
        helpers::applyTypicalPFilter(output, typicalP, block.launchContext());
    }
    return sd::Status::OK;
}

DECLARE_TYPES(typical_p_filter) {
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(typical_p_filter) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

/**
 * xtc_filter - Exclude Top Choices (XTC) logit filter. With probability xtcProbability,
 * among tokens whose softmax probability >= xtcThreshold, mask all EXCEPT the
 * lowest-probability surviving one; otherwise leave logits unchanged. Stochastic (seeded).
 * Float args: 0: xtcProbability (0.0 = off), 1: xtcThreshold (must be < 0.5)
 * Int args:   0: seed
 */
CUSTOM_OP_IMPL(xtc_filter, 1, 1, false, 2, 1) {
    auto logits = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    auto logitsRank = logits->rankOf();
    REQUIRE_TRUE(logitsRank == 1 || logitsRank == 2, 0,
                 "xtc_filter: logits must be rank 1 or 2, got %lld", (long long)logitsRank);

    double xtcProbability = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;
    double xtcThreshold   = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.1;
    sd::LongType seed     = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    // No threshold rejection: xtcThreshold >= 0.5 is a graceful no-op (at most one token can
    // have p >= 0.5, so fewer than two qualify and the kernel leaves logits unchanged).

    if (logits != output) output->assign(logits);

    if (xtcProbability > 0.0) {
        helpers::applyXtcFilter(output, xtcProbability, xtcThreshold, seed, block.launchContext());
    }
    return sd::Status::OK;
}

DECLARE_TYPES(xtc_filter) {
    getOpDescriptor()->addTraits(OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(xtc_filter) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
