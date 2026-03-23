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

    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    NDArray* stateIn = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    int activation = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    helpers::causalConv1d(block.launchContext(), x, weight, bias, stateIn, output, stateOut, activation);

    return sd::Status::OK;
}

DECLARE_TYPES(causal_conv1d) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(causal_conv1d) {
    auto xShape = inputShape->at(0);       // [B, L, D]
    auto weightShape = inputShape->at(1);  // [D, K]

    auto B = shape::sizeAt(xShape, 0);
    auto L = shape::sizeAt(xShape, 1);
    auto D = shape::sizeAt(xShape, 2);
    auto K = shape::sizeAt(weightShape, 1);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, L, D});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, D, K - 1});

    return SHAPELIST(outputShape, stateShape);
}

}  // namespace ops
}  // namespace sd

#endif
