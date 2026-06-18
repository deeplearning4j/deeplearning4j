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

#if NOT_EXCLUDED(OP_moe_shared_experts)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/moe_shared_experts.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(moe_shared_experts, 6, 3, false, -2, 3) {
    auto input               = INPUT_VARIABLE(0);  // [batch, seq_len, hidden_size]
    auto routerWeights       = INPUT_VARIABLE(1);  // [hidden_size, num_routed_experts]
    auto routedExpertWeights = INPUT_VARIABLE(2);  // [num_routed_experts, hidden_size, expert_hidden]
    auto sharedGateProj      = INPUT_VARIABLE(3);  // [hidden_size, shared_intermediate_size]
    auto sharedUpProj        = INPUT_VARIABLE(4);  // [hidden_size, shared_intermediate_size]
    auto sharedDownProj      = INPUT_VARIABLE(5);  // [shared_intermediate_size, hidden_size]
    auto routedExpertBias    = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;

    auto output        = OUTPUT_VARIABLE(0);  // [batch, seq_len, hidden_size]
    auto routerProbs   = OUTPUT_VARIABLE(1);  // [batch, seq_len, num_routed_experts]
    auto expertIndices = OUTPUT_VARIABLE(2);  // [batch, seq_len, top_k]

    auto numRoutedExperts = INT_ARG(0);
    auto topK             = INT_ARG(1);
    auto normalizeProbs   = INT_ARG(2) != 0;

    auto capacityFactor = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    // Validate shapes
    REQUIRE_TRUE(input->rankOf() == 3, 0,
        "moe_shared_experts: input must be rank 3 [batch, seq_len, hidden_size], got rank %i", input->rankOf());
    REQUIRE_TRUE(routerWeights->rankOf() == 2, 0,
        "moe_shared_experts: routerWeights must be rank 2 [hidden_size, num_experts], got rank %i", routerWeights->rankOf());
    REQUIRE_TRUE(routedExpertWeights->rankOf() == 3, 0,
        "moe_shared_experts: routedExpertWeights must be rank 3, got rank %i", routedExpertWeights->rankOf());
    REQUIRE_TRUE(sharedGateProj->rankOf() == 2, 0,
        "moe_shared_experts: sharedGateProj must be rank 2, got rank %i", sharedGateProj->rankOf());
    REQUIRE_TRUE(sharedUpProj->rankOf() == 2, 0,
        "moe_shared_experts: sharedUpProj must be rank 2, got rank %i", sharedUpProj->rankOf());
    REQUIRE_TRUE(sharedDownProj->rankOf() == 2, 0,
        "moe_shared_experts: sharedDownProj must be rank 2, got rank %i", sharedDownProj->rankOf());
    REQUIRE_TRUE(topK > 0 && topK <= numRoutedExperts, 0,
        "moe_shared_experts: topK (%i) must be in [1, numRoutedExperts=%i]", topK, numRoutedExperts);

    if (routedExpertBias != nullptr && routedExpertBias->isEmpty()) {
        routedExpertBias = nullptr;
    }

    helpers::moeSharedExperts(block.launchContext(), input, routerWeights, routedExpertWeights,
                              sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias,
                              output, routerProbs, expertIndices,
                              numRoutedExperts, topK, normalizeProbs, capacityFactor);

    return sd::Status::OK;
}

DECLARE_TYPES(moe_shared_experts) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->setAllowedOutputTypes(1, {ALL_FLOATS})
        ->setAllowedOutputTypes(2, {ALL_INTS});
}

DECLARE_SHAPE_FN(moe_shared_experts) {
    auto inShape = inputShape->at(0);
    auto inputType = sd::ArrayOptions::dataType(inShape);

    auto batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
    auto seqLen    = shape::sizeAt(inShape, static_cast<sd::LongType>(1));
    auto hiddenDim = shape::sizeAt(inShape, static_cast<sd::LongType>(2));

    auto numRoutedExperts = INT_ARG(0);
    auto topK             = INT_ARG(1);

    // Output 0: [batch, seq_len, hidden_size]
    std::vector<sd::LongType> outShape = {batchSize, seqLen, hiddenDim};
    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        inputType, 'c', outShape);

    // Output 1: router_probs [batch, seq_len, num_routed_experts]
    std::vector<sd::LongType> probsShape = {batchSize, seqLen, static_cast<sd::LongType>(numRoutedExperts)};
    auto probsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        inputType, 'c', probsShape);

    // Output 2: expert_indices [batch, seq_len, top_k]
    std::vector<sd::LongType> indicesShape = {batchSize, seqLen, static_cast<sd::LongType>(topK)};
    auto indicesShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        sd::DataType::INT64, 'c', indicesShape);

    return SHAPELIST(outputShapeInfo, probsShapeInfo, indicesShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
