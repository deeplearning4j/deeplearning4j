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
#if defined(SD_CUDA)
        helpers::applyLogitPenaltiesCuda(output, inputIds, repPenalty, freqPenalty, presPenalty,
                                          block.launchContext());
#else
        helpers::applyLogitPenaltiesCpu(output, inputIds, repPenalty, freqPenalty, presPenalty,
                                         block.launchContext());
#endif
    }

    // Apply min-P filtering
    if (minP > 0.0) {
#if defined(SD_CUDA)
        helpers::applyMinPFilterCuda(output, minP, block.launchContext());
#else
        helpers::applyMinPFilterCpu(output, minP, block.launchContext());
#endif
    }

    return sd::Status::OK;
}

DECLARE_TYPES(sampling_penalties) {
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(1, {INT64, INT32});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(sampling_penalties) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
