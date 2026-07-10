/* ******************************************************************************
 *
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
// causal_conv1d - Depthwise causal 1D convolution with state
//
// Performs a causal (left-padded) depthwise 1D convolution.
// Used in Gated Delta Networks (GDN) and Mamba architectures.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_causal_conv1d)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/causal_conv1d.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(causal_conv1d, 2, 2, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);         // [B, L, D]
    auto weight = INPUT_VARIABLE(1);    // [D, K]
    auto output = OUTPUT_VARIABLE(0);   // [B, L, D]
    auto stateOut = OUTPUT_VARIABLE(1); // [B, D, K-1]

    NDArray* bias = nullptr;
    NDArray* stateIn = nullptr;
    NDArray* actualLen = nullptr;

    for (int i = 2; i < block.width(); ++i) {
        auto input = INPUT_VARIABLE(i);
        if (input->rankOf() == 0) {
            actualLen = input;
        } else if (input->rankOf() == 1) {
            bias = input;
        } else {
            stateIn = input;
        }
    }

    REQUIRE_TRUE(actualLen == nullptr || actualLen->dataType() == DataType::INT64, 0,
                 "causal_conv1d: actualLen input must be INT64 scalar");

    int activation = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    helpers::causalConv1d(block.launchContext(), x, weight, bias, stateIn, actualLen,
                          output, stateOut, activation, wFormat);

    return sd::Status::OK;
}

DECLARE_TYPES(causal_conv1d) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(causal_conv1d) {
    auto xShape = inputShape->at(0);       // [B, L, D]
    auto weightShape = inputShape->at(1);  // [D, K]

    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    auto B = shape::sizeAt(xShape, 0);
    auto L = shape::sizeAt(xShape, 1);
    auto D = shape::sizeAt(xShape, 2);
    auto K = (wFormat == 0) ? shape::sizeAt(weightShape, 1) : shape::sizeAt(weightShape, 0);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, L, D});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, D, K - 1});

    return SHAPELIST(outputShape, stateShape);
}

}  // namespace ops
}  // namespace sd

#endif
