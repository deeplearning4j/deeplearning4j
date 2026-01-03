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
// Windowed/Local Attention for efficient transformers
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_windowed_attention)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/windowed_attention.h>

namespace sd {
namespace ops {

/**
 * windowed_attention - Windowed/Local Attention
 *
 * Implements windowed attention mechanisms used in efficient transformers
 * like Longformer, BigBird, Swin Transformer, and SAM.
 *
 * Inputs:
 *   0: query [batch, seq_len, num_heads, head_dim]
 *   1: key [batch, seq_len, num_heads, head_dim]
 *   2: value [batch, seq_len, num_heads, head_dim]
 *   3: relativePositionBias [num_heads, window_size, window_size] (optional)
 *   4: attentionMask [batch, seq_len, seq_len] (optional)
 *
 * Integer args:
 *   0: windowSize - Size of attention window
 *   1: numHeads - Number of attention heads
 *   2: shiftSize - Shift size for shifted window attention (0 for no shift)
 *
 * Float args:
 *   0: scale - Attention scale factor (0 for auto = 1/sqrt(head_dim))
 *
 * Boolean args:
 *   0: returnWeights - Whether to return attention weights
 *
 * Outputs:
 *   0: output [batch, seq_len, num_heads, head_dim]
 *   1: attentionWeights [batch, num_heads, seq_len, seq_len] (if returnWeights)
 */
CUSTOM_OP_IMPL(windowed_attention, 3, -1, false, 1, 3) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto relativePositionBias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto attentionMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

    auto output = OUTPUT_VARIABLE(0);
    auto attentionWeights = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;

    // Get arguments
    auto windowSize = INT_ARG(0);
    auto numHeads = INT_ARG(1);
    auto shiftSize = INT_ARG(2);
    auto scale = T_ARG(0);
    auto returnWeights = block.numB() > 0 ? B_ARG(0) : false;

    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "windowed_attention: query must be rank 4 [batch, seq, heads, dim], got %i", query->rankOf());
    REQUIRE_TRUE(key->rankOf() == 4, 0,
                 "windowed_attention: key must be rank 4, got %i", key->rankOf());
    REQUIRE_TRUE(value->rankOf() == 4, 0,
                 "windowed_attention: value must be rank 4, got %i", value->rankOf());

    // Validate shapes match
    REQUIRE_TRUE(query->isSameShape(key), 0,
                 "windowed_attention: query and key shapes must match");
    REQUIRE_TRUE(query->isSameShape(value), 0,
                 "windowed_attention: query and value shapes must match");

    // Handle empty optional inputs
    if (relativePositionBias != nullptr && relativePositionBias->isEmpty()) {
        relativePositionBias = nullptr;
    }
    if (attentionMask != nullptr && attentionMask->isEmpty()) {
        attentionMask = nullptr;
    }

    // Use helper for CPU/GPU implementation
    helpers::windowedAttention(block.launchContext(),
                                query, key, value, relativePositionBias, attentionMask,
                                output, attentionWeights,
                                windowSize, numHeads, shiftSize, scale, returnWeights);

    return sd::Status::OK;
}

DECLARE_TYPES(windowed_attention) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(windowed_attention) {
    auto queryShape = inputShape->at(0);
    auto inputType = sd::ArrayOptions::dataType(queryShape);

    auto batchSize = shape::sizeAt(queryShape, static_cast<sd::LongType>(0));
    auto seqLen = shape::sizeAt(queryShape, static_cast<sd::LongType>(1));
    auto numHeads = shape::sizeAt(queryShape, static_cast<sd::LongType>(2));
    auto headDim = shape::sizeAt(queryShape, static_cast<sd::LongType>(3));

    auto returnWeights = block.numB() > 0 ? B_ARG(0) : false;

    // Output: same shape as query [batch, seq, heads, dim]
    std::vector<sd::LongType> outputShape = {batchSize, seqLen, numHeads, headDim};
    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outputShape);

    if (returnWeights) {
        // Attention weights: [batch, num_heads, seq_len, seq_len]
        std::vector<sd::LongType> weightsShape = {batchSize, numHeads, seqLen, seqLen};
        auto weightsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', weightsShape);
        return SHAPELIST(outputShapeInfo, weightsShapeInfo);
    }

    return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
