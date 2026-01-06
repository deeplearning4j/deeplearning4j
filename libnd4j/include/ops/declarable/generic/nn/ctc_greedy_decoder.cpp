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

//
// @author Eclipse Deeplearning4j
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_ctc_greedy_decoder)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/ctc_decode.h>

namespace sd {
namespace ops {

/**
 * ctc_greedy_decoder - CTC Greedy Decoder for OCR and speech recognition
 *
 * Performs greedy decoding on CTC output by selecting the most likely
 * token at each timestep and optionally merging repeated tokens.
 *
 * Inputs:
 *   0: logits [batch, time, num_classes] - raw network outputs or log probabilities
 *   1: sequence_length [batch] (optional) - actual sequence lengths
 *
 * Integer args:
 *   0: merge_repeated - whether to merge consecutive identical tokens (default 1)
 *   1: blank_index - index of blank token (default -1, meaning num_classes-1)
 *
 * Outputs:
 *   0: decoded [batch, time] - decoded token indices (invalid positions filled with -1)
 *   1: log_probability [batch] - log probability of decoded sequence
 */
CUSTOM_OP_IMPL(ctc_greedy_decoder, 1, 2, false, 0, 2) {
    auto logits = INPUT_VARIABLE(0);
    auto sequenceLength = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;

    auto decoded = OUTPUT_VARIABLE(0);
    auto logProb = OUTPUT_VARIABLE(1);

    // Get arguments
    auto mergeRepeated = INT_ARG(0) != 0;
    auto blankIndex = INT_ARG(1);

    REQUIRE_TRUE(logits->rankOf() == 3, 0,
                 "ctc_greedy_decoder: logits must be rank 3 [batch, time, classes], got rank %i",
                 logits->rankOf());

    auto numClasses = logits->sizeAt(2);
    REQUIRE_TRUE(blankIndex < 0 || (blankIndex >= 0 && blankIndex < numClasses), 0,
                 "ctc_greedy_decoder: blank_index must be in range [0, num_classes), got %i with num_classes=%lld",
                 blankIndex, numClasses);

    // Initialize decoded output to -1 (invalid)
    sd::LongType initVal = -1;
    decoded->assign(initVal);

    // Use helper for CPU/GPU implementation
    helpers::ctcGreedyDecode(block.launchContext(), logits, sequenceLength,
                              decoded, logProb, blankIndex, mergeRepeated);

    return sd::Status::OK;
}

DECLARE_TYPES(ctc_greedy_decoder) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedOutputTypes(0, {ALL_INTS})
        ->setAllowedOutputTypes(1, {ALL_FLOATS});
}

DECLARE_SHAPE_FN(ctc_greedy_decoder) {
    auto logitsShape = inputShape->at(0);

    auto batchSize = shape::sizeAt(logitsShape, static_cast<sd::LongType>(0));
    auto timeSteps = shape::sizeAt(logitsShape, static_cast<sd::LongType>(1));

    // Output shapes
    // decoded: [batch, time] - INT64
    // logProb: [batch] - same float type as input
    std::vector<sd::LongType> decodedShape = {batchSize, timeSteps};
    std::vector<sd::LongType> logProbShape = {batchSize};

    auto decodedShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        sd::DataType::INT64, 'c', decodedShape);
    auto logProbShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(logitsShape), 'c', logProbShape);

    return SHAPELIST(decodedShapeInfo, logProbShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
