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
#if NOT_EXCLUDED(OP_fp8_dequantize)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/per_token_quant.h>

namespace sd {
namespace ops {

/**
 * fp8_dequantize - Dequantize FP8 E4M3 tensor back to float.
 *
 * Input:
 *   0: quantized tensor [numTokens, hiddenDim] INT8 (FP8 E4M3 values)
 *   1: scale factors [numTokens] or [1] float32
 *
 * Output:
 *   0: dequantized tensor [numTokens, hiddenDim] - float/half
 *
 * Integer args:
 *   0: outputType (0=FLOAT32, 1=FLOAT16, default: 0)
 */
CUSTOM_OP_IMPL(fp8_dequantize, 2, 1, false, 0, 0) {
    auto quantized = INPUT_VARIABLE(0);
    auto scales = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(quantized->rankOf() == 2, 0,
                 "fp8_dequantize: quantized input must be rank 2 [numTokens, hiddenDim], got %lld",
                 (long long)quantized->rankOf());

    helpers::perTokenDequantFp8(block.launchContext(), quantized, scales, output);

    return sd::Status::OK;
}

DECLARE_TYPES(fp8_dequantize) {
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes(0, {INT8});
    getOpDescriptor()->setAllowedInputTypes(1, {FLOAT32});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(fp8_dequantize) {
    auto inShape = inputShape->at(0);
    auto numTokens = shape::sizeAt(inShape, static_cast<LongType>(0));
    auto hiddenDim = shape::sizeAt(inShape, static_cast<LongType>(1));

    int outputTypeArg = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    DataType outDtype = (outputTypeArg == 1) ? HALF : FLOAT32;

    LongType shapeArr[2] = {numTokens, hiddenDim};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        outDtype, 'c', 2, shapeArr));
}

}  // namespace ops
}  // namespace sd

#endif
