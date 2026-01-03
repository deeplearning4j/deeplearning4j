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
// Relative Position Bias for attention mechanisms
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_relative_position_bias)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/relative_position_bias.h>

namespace sd {
namespace ops {

/**
 * relative_position_bias - Compute relative position bias for attention
 *
 * Supports two modes:
 * 1. Learned bias (Swin/SAM style): looks up bias values from a learned table
 * 2. ALiBi: computes linear position-based bias
 *
 * Inputs:
 *   0: biasTable [num_relative_positions, num_heads] (for learned mode)
 *      OR sequenceLength scalar/tensor (for ALiBi mode)
 *   1: relativePositionIndex [window_size^2, window_size^2] (optional, for learned mode)
 *
 * Integer args:
 *   0: numHeads - Number of attention heads
 *   1: windowSize - Window size for 2D position encoding (ignored in ALiBi mode)
 *   2: useAlibi - 1 for ALiBi mode, 0 for learned mode
 *
 * Outputs:
 *   0: output [num_heads, seq_len, seq_len] or [num_heads, window_size^2, window_size^2]
 */
CUSTOM_OP_IMPL(relative_position_bias, 1, 1, false, 0, 3) {
    auto biasTable = INPUT_VARIABLE(0);
    auto relativePositionIndex = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;

    auto output = OUTPUT_VARIABLE(0);

    // Get arguments
    auto numHeads = INT_ARG(0);
    auto windowSize = INT_ARG(1);
    auto useAlibi = INT_ARG(2) != 0;

    // Handle empty optional inputs
    if (relativePositionIndex != nullptr && relativePositionIndex->isEmpty()) {
        relativePositionIndex = nullptr;
    }

    // Validate inputs based on mode
    if (!useAlibi) {
        REQUIRE_TRUE(biasTable->rankOf() == 2, 0,
                     "relative_position_bias: biasTable must be rank 2 [num_rel_pos, num_heads], got %i",
                     biasTable->rankOf());
        REQUIRE_TRUE(biasTable->sizeAt(1) == numHeads, 0,
                     "relative_position_bias: biasTable second dimension must equal numHeads");
    }

    // Use helper for CPU/GPU implementation
    helpers::relativePositionBias(block.launchContext(),
                                   biasTable, relativePositionIndex, output,
                                   numHeads, windowSize, useAlibi);

    return sd::Status::OK;
}

DECLARE_TYPES(relative_position_bias) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(relative_position_bias) {
    auto inShape = inputShape->at(0);
    auto inputType = sd::ArrayOptions::dataType(inShape);

    auto numHeads = INT_ARG(0);
    auto windowSize = INT_ARG(1);
    auto useAlibi = INT_ARG(2) != 0;

    // Determine output type
    sd::DataType outputType = inputType;
    if (!sd::DataTypeUtils::isR(inputType)) {
        outputType = sd::DataType::FLOAT32;
    }

    std::vector<sd::LongType> outShape;
    if (useAlibi) {
        // For ALiBi, infer sequence length from input
        // If input is a scalar, use it as sequence length
        // Otherwise, use a reasonable default or shape inference
        sd::LongType seqLen = 512;  // Default
        if (shape::isScalar(inShape)) {
            // Will be determined at runtime
        } else if (shape::rank(inShape) >= 1) {
            seqLen = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
        }
        outShape = {numHeads, seqLen, seqLen};
    } else {
        // For learned bias
        sd::LongType windowArea = windowSize * windowSize;
        outShape = {numHeads, windowArea, windowArea};
    }

    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(outputType, 'c', outShape);
    return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
