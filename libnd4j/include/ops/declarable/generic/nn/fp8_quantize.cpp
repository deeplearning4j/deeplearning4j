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
#if NOT_EXCLUDED(OP_fp8_quantize)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/per_token_quant.h>

namespace sd {
namespace ops {

/**
 * fp8_quantize - Quantize float tensor to FP8 E4M3 representation.
 *
 * Supports two modes:
 *   mode=0: Per-tensor quantization (single scale for entire tensor)
 *   mode=1: Per-token (per-row) quantization (one scale per row)
 *   mode=2: Per-group quantization (one scale per group of groupSize elements)
 *
 * Input:
 *   0: input tensor [numTokens, hiddenDim] - float/half/bf16
 *
 * Output:
 *   0: quantized tensor [numTokens, hiddenDim] - INT8 (stores FP8 E4M3 values)
 *   1: scale factors - [1] for per-tensor, [numTokens] for per-token,
 *                       [numTokens, numGroups] for per-group
 *
 * Integer args:
 *   0: mode (0=per-tensor, 1=per-token, 2=per-group, default: 1)
 *   1: groupSize (for mode=2, default: 128)
 *   2: fp8Format (0=E4M3, 1=E5M2, default: 0)
 */
CUSTOM_OP_IMPL(fp8_quantize, 1, 2, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto quantized = OUTPUT_VARIABLE(0);
    auto scales = OUTPUT_VARIABLE(1);

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int groupSize = block.getIArguments()->size() > 1 ? INT_ARG(1) : 128;

    REQUIRE_TRUE(input->rankOf() == 2, 0,
                 "fp8_quantize: input must be rank 2 [numTokens, hiddenDim], got %lld",
                 (long long)input->rankOf());

    if (mode == 2) {
        // Per-group
        helpers::perTokenGroupQuant(block.launchContext(), input, quantized, scales, groupSize, true);
    } else {
        // Per-token (mode 0 and 1 both use perTokenQuantFp8;
        // for mode=0/per-tensor, the caller reshapes to [1, totalElements])
        helpers::perTokenQuantFp8(block.launchContext(), input, quantized, scales);
    }

    return sd::Status::OK;
}

DECLARE_TYPES(fp8_quantize) {
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes(0, {INT8});
    getOpDescriptor()->setAllowedOutputTypes(1, {FLOAT32});
}

DECLARE_SHAPE_FN(fp8_quantize) {
    auto inShape = inputShape->at(0);
    auto numTokens = shape::sizeAt(inShape, static_cast<LongType>(0));
    auto hiddenDim = shape::sizeAt(inShape, static_cast<LongType>(1));

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int groupSize = block.getIArguments()->size() > 1 ? INT_ARG(1) : 128;

    // Output 0: quantized [numTokens, hiddenDim] INT8
    LongType quantShape[2] = {numTokens, hiddenDim};
    auto quantizedShape = ConstantShapeHelper::getInstance().createShapeInfo(
        INT8, 'c', 2, quantShape);

    // Output 1: scales
    LongType* scalesShape;
    if (mode == 2) {
        // Per-group: [numTokens, numGroups]
        LongType numGroups = (hiddenDim + groupSize - 1) / groupSize;
        LongType groupShape[2] = {numTokens, numGroups};
        scalesShape = ConstantShapeHelper::getInstance().createShapeInfo(
            FLOAT32, 'c', 2, groupShape);
    } else {
        // Per-token: [numTokens]
        scalesShape = ConstantShapeHelper::getInstance().vectorShapeInfo(numTokens, FLOAT32);
    }

    return SHAPELIST(quantizedShape, scalesShape);
}

}  // namespace ops
}  // namespace sd

#endif
