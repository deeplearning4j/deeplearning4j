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
#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/causal_conv1d.h>

#if NOT_EXCLUDED(OP_causal_conv1d)

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(causal_conv1d, 2, 2, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);         // [B, L, D]
    auto weight = INPUT_VARIABLE(1);    // [D, K]
    auto output = OUTPUT_VARIABLE(0);   // [B, L, D]
    auto stateOut = OUTPUT_VARIABLE(1); // [B, D, K-1]

    helpers::CausalConv1dInputRoles inputRoles;
    std::string resolutionFailure;
    const bool inputsResolved = helpers::resolveCausalConv1dInputRoles(
        block.width(),
        [&](int inputIndex) { return INPUT_VARIABLE(inputIndex)->rankOf(); },
        inputRoles, &resolutionFailure);
    REQUIRE_TRUE(inputsResolved, 0, "causal_conv1d: invalid input contract: %s",
                 resolutionFailure.c_str());

    NDArray* bias = inputRoles.bias >= 0 ? INPUT_VARIABLE(inputRoles.bias) : nullptr;
    NDArray* stateIn = inputRoles.stateIn >= 0 ? INPUT_VARIABLE(inputRoles.stateIn) : nullptr;
    NDArray* actualLen =
        inputRoles.actualLen >= 0 ? INPUT_VARIABLE(inputRoles.actualLen) : nullptr;

    REQUIRE_TRUE(actualLen == nullptr || actualLen->dataType() == DataType::INT64, 0,
                 "causal_conv1d: actualLen input must be INT64 scalar");
    REQUIRE_TRUE(bias == nullptr || bias->dataType() == x->dataType(), 0,
                 "causal_conv1d: bias dtype must match activation dtype");

    int activation = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    helpers::causalConv1d(block.launchContext(), x, weight, bias, stateIn, actualLen,
                          output, stateOut, activation, wFormat);

    return sd::Status::OK;
}

DECLARE_TYPES(causal_conv1d) {
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(causal_conv1d) {
    auto xShape = inputShape->at(0);       // [B, L, D]
    auto weightShape = inputShape->at(1);  // [D, K]

    helpers::CausalConv1dInputRoles inputRoles;
    std::string resolutionFailure;
    const bool inputsResolved = helpers::resolveCausalConv1dInputRoles(
        block.width(),
        [&](int inputIndex) { return shape::rank(inputShape->at(inputIndex)); },
        inputRoles, &resolutionFailure);
    REQUIRE_TRUE(inputsResolved, 0, "causal_conv1d: invalid input contract: %s",
                 resolutionFailure.c_str());

    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    auto B = shape::sizeAt(xShape, 0);
    auto L = shape::sizeAt(xShape, 1);
    auto D = shape::sizeAt(xShape, 2);
    auto K = (wFormat == 0) ? shape::sizeAt(weightShape, 1) : shape::sizeAt(weightShape, 0);
    const auto stateType = inputRoles.stateIn >= 0
        ? ArrayOptions::dataType(inputShape->at(inputRoles.stateIn))
        : ArrayOptions::dataType(xShape);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, L, D});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        stateType, 'c', {B, D, K - 1});

    return SHAPELIST(outputShape, stateShape);
}

}  // namespace ops
}  // namespace sd

#endif

#if NOT_EXCLUDED(OP_causal_conv1d_with_prefix)

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(causal_conv1d_with_prefix, 2, 3, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);          // [B, L, D]
    auto weight = INPUT_VARIABLE(1);     // [D, K]
    auto output = OUTPUT_VARIABLE(0);    // [B, L, D]
    auto stateOut = OUTPUT_VARIABLE(1);  // [B, D, K-1]
    auto prefixOut = OUTPUT_VARIABLE(2); // [W, B, D, K-1] time-leading history checkpoints

    helpers::CausalConv1dInputRoles inputRoles;
    std::string resolutionFailure;
    const bool inputsResolved = helpers::resolveCausalConv1dInputRoles(
        block.width(),
        [&](int inputIndex) { return INPUT_VARIABLE(inputIndex)->rankOf(); },
        inputRoles, &resolutionFailure);
    REQUIRE_TRUE(inputsResolved, 0, "causal_conv1d_with_prefix: invalid input contract: %s",
                 resolutionFailure.c_str());

    NDArray* bias = inputRoles.bias >= 0 ? INPUT_VARIABLE(inputRoles.bias) : nullptr;
    NDArray* stateIn = inputRoles.stateIn >= 0 ? INPUT_VARIABLE(inputRoles.stateIn) : nullptr;
    NDArray* actualLen =
        inputRoles.actualLen >= 0 ? INPUT_VARIABLE(inputRoles.actualLen) : nullptr;

    // Prefix capture requires explicit length masking.
    REQUIRE_TRUE(actualLen != nullptr, 0,
                 "causal_conv1d_with_prefix: an INT64 actualLen scalar input is required");
    REQUIRE_TRUE(actualLen->dataType() == DataType::INT64, 0,
                 "causal_conv1d_with_prefix: actualLen input must be INT64 scalar");
    REQUIRE_TRUE(bias == nullptr || bias->dataType() == x->dataType(), 0,
                 "causal_conv1d_with_prefix: bias dtype must match activation dtype");
    const auto stateType = stateIn != nullptr ? stateIn->dataType() : x->dataType();
    REQUIRE_TRUE(prefixOut->dataType() == stateType, 0,
                 "causal_conv1d_with_prefix: prefixOut dtype must match state dtype");
    REQUIRE_TRUE(prefixOut->rankOf() == 4 &&
                     prefixOut->sizeAt(1) == x->sizeAt(0) &&
                     prefixOut->sizeAt(2) == x->sizeAt(2) &&
                     prefixOut->sizeAt(3) == stateOut->sizeAt(2), 0,
                 "causal_conv1d_with_prefix: prefixOut must have shape [W,B,D,K-1]");

    int activation = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    helpers::causalConv1dWithPrefix(block.launchContext(), x, weight, bias, stateIn, actualLen,
                                    output, stateOut, prefixOut, activation, wFormat);

    return sd::Status::OK;
}

DECLARE_TYPES(causal_conv1d_with_prefix) {
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(causal_conv1d_with_prefix) {
    auto xShape = inputShape->at(0);       // [B, L, D]
    auto weightShape = inputShape->at(1);  // [D, K]

    helpers::CausalConv1dInputRoles inputRoles;
    std::string resolutionFailure;
    const bool inputsResolved = helpers::resolveCausalConv1dInputRoles(
        block.width(),
        [&](int inputIndex) { return shape::rank(inputShape->at(inputIndex)); },
        inputRoles, &resolutionFailure);
    REQUIRE_TRUE(inputsResolved, 0, "causal_conv1d_with_prefix: invalid input contract: %s",
                 resolutionFailure.c_str());

    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    auto B = shape::sizeAt(xShape, 0);
    auto L = shape::sizeAt(xShape, 1);
    auto D = shape::sizeAt(xShape, 2);
    auto K = (wFormat == 0) ? shape::sizeAt(weightShape, 1) : shape::sizeAt(weightShape, 0);
    const auto stateType = inputRoles.stateIn >= 0
        ? ArrayOptions::dataType(inputShape->at(inputRoles.stateIn))
        : ArrayOptions::dataType(xShape);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, L, D});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        stateType, 'c', {B, D, K - 1});

    // Physical prefix capacity W = L at shape time (verification window width).
    auto prefixShape = ConstantShapeHelper::getInstance().createShapeInfo(
        stateType, 'c', {L, B, D, K - 1});

    return SHAPELIST(outputShape, stateShape, prefixShape);
}

}  // namespace ops
}  // namespace sd

#endif
