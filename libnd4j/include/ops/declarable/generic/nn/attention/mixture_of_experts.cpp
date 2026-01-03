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
// Sparse Mixture of Experts for efficient large models
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_mixture_of_experts)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/mixture_of_experts.h>

namespace sd {
namespace ops {

/**
 * mixture_of_experts - Sparse Mixture of Experts layer
 *
 * Implements sparse MoE used in models like Mixtral, Switch Transformer, etc.
 * Routes each token to top-K experts and combines their outputs.
 *
 * Inputs:
 *   0: input [batch, seq_len, hidden_dim]
 *   1: gatingWeights [hidden_dim, num_experts] - router/gating network weights
 *   2: expertWeights [num_experts, hidden_dim, expert_dim] - expert network weights
 *   3: expertBias [num_experts, expert_dim] (optional)
 *
 * Integer args:
 *   0: numExperts - Total number of experts
 *   1: topK - Number of experts to route to per token
 *   2: expertCapacity - Maximum tokens per expert (0 for unlimited)
 *
 * Boolean args:
 *   0: useSoftmaxGating - Whether to apply softmax to gating (default true)
 *   1: returnRouterLogits - Whether to return router logits (default false)
 *
 * Outputs:
 *   0: output [batch, seq_len, hidden_dim]
 *   1: routerLogits [batch, seq_len, num_experts] (if returnRouterLogits)
 */
CUSTOM_OP_IMPL(mixture_of_experts, 3, -1, false, 0, 3) {
    auto input = INPUT_VARIABLE(0);
    auto gatingWeights = INPUT_VARIABLE(1);
    auto expertWeights = INPUT_VARIABLE(2);
    auto expertBias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    auto output = OUTPUT_VARIABLE(0);
    auto routerLogits = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;

    // Get arguments
    auto numExperts = INT_ARG(0);
    auto topK = INT_ARG(1);
    auto expertCapacity = INT_ARG(2);
    auto useSoftmaxGating = block.numB() > 0 ? B_ARG(0) : true;
    auto returnRouterLogits = block.numB() > 1 ? B_ARG(1) : false;

    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "mixture_of_experts: input must be rank 3 [batch, seq, hidden], got %i", input->rankOf());
    REQUIRE_TRUE(gatingWeights->rankOf() == 2, 0,
                 "mixture_of_experts: gatingWeights must be rank 2, got %i", gatingWeights->rankOf());
    REQUIRE_TRUE(expertWeights->rankOf() == 3, 0,
                 "mixture_of_experts: expertWeights must be rank 3, got %i", expertWeights->rankOf());

    REQUIRE_TRUE(topK > 0 && topK <= numExperts, 0,
                 "mixture_of_experts: topK must be in range [1, numExperts], got %i", topK);

    // Handle empty optional inputs
    if (expertBias != nullptr && expertBias->isEmpty()) {
        expertBias = nullptr;
    }

    // Use helper for CPU/GPU implementation
    helpers::mixtureOfExpertsSimple(block.launchContext(),
                                     input, gatingWeights, expertWeights, expertBias,
                                     output, numExperts, topK);

    return sd::Status::OK;
}

DECLARE_TYPES(mixture_of_experts) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(mixture_of_experts) {
    auto inShape = inputShape->at(0);
    auto inputType = sd::ArrayOptions::dataType(inShape);

    auto batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
    auto seqLen = shape::sizeAt(inShape, static_cast<sd::LongType>(1));
    auto hiddenDim = shape::sizeAt(inShape, static_cast<sd::LongType>(2));

    auto numExperts = INT_ARG(0);
    auto returnRouterLogits = block.numB() > 1 ? B_ARG(1) : false;

    // Output: same shape as input [batch, seq, hidden]
    std::vector<sd::LongType> outShape = {batchSize, seqLen, hiddenDim};
    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outShape);

    if (returnRouterLogits) {
        // Router logits: [batch, seq, num_experts]
        std::vector<sd::LongType> logitsShape = {batchSize, seqLen, numExperts};
        auto logitsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', logitsShape);
        return SHAPELIST(outputShapeInfo, logitsShapeInfo);
    }

    return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
