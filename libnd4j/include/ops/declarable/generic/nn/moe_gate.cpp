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

#if NOT_EXCLUDED(OP_moe_gate)

#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/activations.h>
#include <ops/declarable/helpers/one_hot.h>
#include <ops/declarable/helpers/top_k.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(moe_gate, 2, 3, false, -2, -2) {
    auto hiddenStates = INPUT_VARIABLE(0);  // [tokens, hidden]
    auto gateWeight   = INPUT_VARIABLE(1);  // [hidden, numExperts]

    auto expertIndices = OUTPUT_VARIABLE(0);  // [tokens, topK] INT64
    auto gateWeights   = OUTPUT_VARIABLE(1);  // [tokens, topK]
    auto auxLoss       = OUTPUT_VARIABLE(2);  // [1]

    const LongType topK       = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    const LongType numExperts = block.getIArguments()->size() > 1 ? INT_ARG(1) : gateWeight->sizeAt(1);
    const double auxLossCoeff = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.01;

    REQUIRE_TRUE(hiddenStates->rankOf() == 2, 0,
                 "moe_gate: hiddenStates must be rank 2 [tokens, hidden], got rank %i", hiddenStates->rankOf());
    REQUIRE_TRUE(gateWeight->rankOf() == 2, 0,
                 "moe_gate: gateWeight must be rank 2 [hidden, numExperts], got rank %i", gateWeight->rankOf());
    REQUIRE_TRUE(gateWeight->sizeAt(0) == hiddenStates->sizeAt(1), 0,
                 "moe_gate: gateWeight rows (%lld) must match hiddenStates hidden dim (%lld)",
                 (long long)gateWeight->sizeAt(0), (long long)hiddenStates->sizeAt(1));
    REQUIRE_TRUE(gateWeight->sizeAt(1) == numExperts, 0,
                 "moe_gate: gateWeight columns (%lld) must equal numExperts (%lld)",
                 (long long)gateWeight->sizeAt(1), (long long)numExperts);
    REQUIRE_TRUE(topK > 0 && topK <= numExperts, 0,
                 "moe_gate: topK (%lld) must be in [1, numExperts=%lld]", (long long)topK, (long long)numExperts);

    if (hiddenStates->isEmpty()) {
        auxLoss->p(0, 0.0);
        return Status::OK;
    }

    const LongType numTokens = hiddenStates->sizeAt(0);

    // router logits [tokens, numExperts], then softmax over the expert axis.
    // softmax needs a distinct output buffer (stateful transform).
    std::vector<sd::LongType> logitsShape = {numTokens, numExperts};
    NDArray logits('c', logitsShape, hiddenStates->dataType(), block.launchContext());
    NDArray probs('c', logitsShape, hiddenStates->dataType(), block.launchContext());
    MmulHelper::mmul(hiddenStates, gateWeight, &logits, 1.0, 0.0);
    helpers::softmax(block.launchContext(), &logits, &probs, 1);

    // top-K expert selection per token
    std::vector<sd::LongType> selShape = {numTokens, topK};
    NDArray topVals('c', selShape, hiddenStates->dataType(), block.launchContext());
    auto tkStatus = helpers::topKFunctor(block.launchContext(), &probs, &topVals, expertIndices, topK, true);
    REQUIRE_TRUE(tkStatus == Status::OK, 0, "moe_gate: top-K selection failed");

    // renormalize the selected probabilities so each token's weights sum to 1
    std::vector<LongType> lastDim = {1};
    NDArray* rowSum = topVals.reduceAlongDimension(reduce::Sum, &lastDim, true);  // [tokens, 1]
    topVals.applyTrueBroadcast(BroadcastOpsTuple::Divide(), rowSum, gateWeights, true);

    // load-balancing auxiliary loss: coeff * E * sum_e(meanProb_e * routedFrac_e)
    std::vector<LongType> axis0 = {0};
    NDArray* meanProbs = probs.reduceAlongDimension(reduce::Mean, &axis0, false);  // [numExperts]

    std::vector<sd::LongType> oneHotShape = {numTokens, topK, numExperts};
    NDArray oneHot('c', oneHotShape, hiddenStates->dataType(), block.launchContext());
    helpers::onehot(block.launchContext(), expertIndices, &oneHot, 2, numExperts, 1.0, 0.0);
    std::vector<LongType> axes01 = {0, 1};
    NDArray* routedFrac = oneHot.reduceAlongDimension(reduce::Mean, &axes01, false);  // [numExperts]

    *meanProbs *= *routedFrac;
    NDArray* balance = meanProbs->reduceNumber(reduce::Sum);
    auxLoss->p(0, auxLossCoeff * static_cast<double>(numExperts) * balance->e<double>(0));

    delete rowSum;
    delete meanProbs;
    delete routedFrac;
    delete balance;

    return Status::OK;
}

DECLARE_TYPES(moe_gate) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedOutputTypes(0, {ALL_INTS})
        ->setAllowedOutputTypes(1, {ALL_FLOATS})
        ->setAllowedOutputTypes(2, {ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(moe_gate) {
    auto hsShape = inputShape->at(0);
    auto inputType = sd::ArrayOptions::dataType(hsShape);

    const LongType numTokens = shape::sizeAt(hsShape, static_cast<sd::LongType>(0));
    const LongType topK = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;

    std::vector<sd::LongType> selShape = {numTokens, topK};
    auto indicesShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::INT64, 'c', selShape);
    auto weightsShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', selShape);

    std::vector<sd::LongType> lossShape = {1};
    auto lossShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', lossShape);

    return SHAPELIST(indicesShapeInfo, weightsShapeInfo, lossShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
