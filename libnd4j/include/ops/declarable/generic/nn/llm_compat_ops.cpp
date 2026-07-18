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
// Native adapters for op names formerly provided only by the (removed)
// platform/llamacpp helpers. Pure delegations/compositions — no kernels.
//

#include <system/op_boilerplate.h>

#include <helpers/ConstantShapeHelper.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/one_hot.h>
#include <ops/declarable/helpers/scatter.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_scale)
CUSTOM_OP_IMPL(scale, 1, 1, false, -2, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const double scaleVal = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    if (input->isEmpty()) return Status::OK;

    input->applyScalar(scalar::Multiply, scaleVal, output);
    return Status::OK;
}

DECLARE_TYPES(scale) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(scale) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_add_inplace)
CUSTOM_OP_IMPL(add_inplace, 2, 1, true, 0, 0) {
    auto accumulator = INPUT_VARIABLE(0);
    auto input = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(accumulator->isSameShape(input), 0,
                 "add_inplace: accumulator and input must have the same shape");

    if (accumulator->isEmpty()) return Status::OK;

    accumulator->applyPairwiseTransform(pairwise::Add, input, output, nullptr);
    return Status::OK;
}

DECLARE_TYPES(add_inplace) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(add_inplace) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_get_rows_bp)
CUSTOM_OP_IMPL(get_rows_bp, 2, 1, false, 0, 1) {
    auto gradOutput = INPUT_VARIABLE(0);  // [N, D]
    auto indices = INPUT_VARIABLE(1);     // [N]
    auto gradWeights = OUTPUT_VARIABLE(0);  // [numRows, D]

    REQUIRE_TRUE(gradOutput->rankOf() == 2, 0,
                 "get_rows_bp: gradOutput must be rank 2 [N, D], got rank %i", gradOutput->rankOf());
    REQUIRE_TRUE(indices->rankOf() == 1, 0,
                 "get_rows_bp: indices must be rank 1 [N], got rank %i", indices->rankOf());
    REQUIRE_TRUE(indices->lengthOf() == gradOutput->sizeAt(0), 0,
                 "get_rows_bp: indices length (%lld) must match gradOutput rows (%lld)",
                 (long long)indices->lengthOf(), (long long)gradOutput->sizeAt(0));

    gradWeights->nullify();
    if (gradOutput->isEmpty() || indices->isEmpty()) return Status::OK;

    // scatter-add rows: gradWeights[indices[i], :] += gradOutput[i, :]
    helpers::scatter(block.launchContext(), pairwise::Add, *indices, *gradOutput, *gradWeights, true);
    return Status::OK;
}

DECLARE_TYPES(get_rows_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

DECLARE_SHAPE_FN(get_rows_bp) {
    auto gradOutShape = inputShape->at(0);
    const LongType numRows = INT_ARG(0);
    REQUIRE_TRUE(numRows > 0, 0, "get_rows_bp: I arg 0 (numRows) must be > 0, got %lld", (long long)numRows);

    std::vector<sd::LongType> outShape = {numRows, shape::sizeAt(gradOutShape, static_cast<sd::LongType>(1))};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(gradOutShape), 'c', outShape));
}
#endif

//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_paged_attention)
CUSTOM_OP_IMPL(paged_attention, 5, 1, false, -2, -2) {
    auto query = INPUT_VARIABLE(0);           // [B, 1, H, Dh]
    auto keyBlockPool = INPUT_VARIABLE(1);    // [numBlocks, blockSize, kvH, Dh]
    auto valueBlockPool = INPUT_VARIABLE(2);  // [numBlocks, blockSize, kvH, Dh]
    auto pageTables = INPUT_VARIABLE(3);      // [B, maxBlocksPerSeq] int32
    auto contextLens = INPUT_VARIABLE(4);     // [B] int32
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(query->rankOf() == 4, 0, "paged_attention: query must be rank 4, got %i", query->rankOf());
    REQUIRE_TRUE(keyBlockPool->rankOf() == 4, 0, "paged_attention: keyBlockPool must be rank 4, got %i",
                 keyBlockPool->rankOf());

    // llama.cpp-era callers passed only blockSize; everything else derives from shapes.
    const LongType blockSize =
        block.getIArguments()->size() > 0 ? INT_ARG(0) : keyBlockPool->sizeAt(1);
    const LongType numHeads = query->sizeAt(2);
    const LongType numKvHeads = keyBlockPool->sizeAt(2);
    const LongType headDim = query->sizeAt(3);
    const double scaleVal = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;  // 0 = auto

    sd::ops::paged_attention_forward inner;
    auto status = inner.execute({query, keyBlockPool, valueBlockPool, pageTables, contextLens}, {output},
                                {scaleVal}, {blockSize, numHeads, numKvHeads, headDim}, {});
    REQUIRE_TRUE(status == Status::OK, 0, "paged_attention: delegation to paged_attention_forward failed");
    return status;
}

DECLARE_TYPES(paged_attention) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {INT32})
        ->setAllowedInputTypes(4, {INT32})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(paged_attention) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

//////////////////////////////////////////////////////////////////////////
#if NOT_EXCLUDED(OP_moe_expert_ffn)
CUSTOM_OP_IMPL(moe_expert_ffn, 4, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);           // [T, H]
    auto expertWeights = INPUT_VARIABLE(1);   // [E, H, D]
    auto routingWeights = INPUT_VARIABLE(2);  // [T, K]
    auto expertIndices = INPUT_VARIABLE(3);   // [T, K] int
    auto output = OUTPUT_VARIABLE(0);         // [T, D]

    REQUIRE_TRUE(input->rankOf() == 2, 0, "moe_expert_ffn: input must be rank 2 [T, H], got %i", input->rankOf());
    REQUIRE_TRUE(expertWeights->rankOf() == 3, 0,
                 "moe_expert_ffn: expertWeights must be rank 3 [E, H, D], got %i", expertWeights->rankOf());
    REQUIRE_TRUE(routingWeights->rankOf() == 2 && expertIndices->isSameShape(routingWeights), 0,
                 "moe_expert_ffn: routingWeights/expertIndices must both be [T, K]");
    REQUIRE_TRUE(expertWeights->sizeAt(1) == input->sizeAt(1), 0,
                 "moe_expert_ffn: expertWeights H dim (%lld) must match input H dim (%lld)",
                 (long long)expertWeights->sizeAt(1), (long long)input->sizeAt(1));

    const LongType numTokens = input->sizeAt(0);
    const LongType hiddenDim = input->sizeAt(1);
    const LongType numExperts =
        block.getIArguments()->size() > 0 ? INT_ARG(0) : expertWeights->sizeAt(0);
    const LongType topK = routingWeights->sizeAt(1);
    REQUIRE_TRUE(numExperts == expertWeights->sizeAt(0), 0,
                 "moe_expert_ffn: numExperts (%lld) must equal expertWeights E dim (%lld)",
                 (long long)numExperts, (long long)expertWeights->sizeAt(0));

    if (input->isEmpty()) {
        output->nullify();
        return Status::OK;
    }

    // Dense routing matrix gateFull[t, e] = sum_k routingWeights[t, k] * 1[expertIndices[t, k] == e].
    std::vector<sd::LongType> oneHotShape = {numTokens, topK, numExperts};
    NDArray oneHot('c', oneHotShape, routingWeights->dataType(), block.launchContext());
    helpers::onehot(block.launchContext(), expertIndices, &oneHot, 2, numExperts, 1.0, 0.0);

    std::vector<sd::LongType> routing3Shape = {numTokens, topK, 1};
    NDArray* routing3 = routingWeights->reshape('c', routing3Shape, true);
    NDArray weighted('c', oneHotShape, routingWeights->dataType(), block.launchContext());
    oneHot.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), routing3, &weighted, true);

    std::vector<sd::LongType> axis1 = {1};
    NDArray* gateFull = weighted.reduceAlongDimension(reduce::Sum, &axis1, false);  // [T, E]

    // xg[t, e, h] = gateFull[t, e] * input[t, h]
    std::vector<sd::LongType> gate3Shape = {numTokens, numExperts, 1};
    std::vector<sd::LongType> input3Shape = {numTokens, 1, hiddenDim};
    NDArray* gate3 = gateFull->reshape('c', gate3Shape, true);
    NDArray* input3 = input->reshape('c', input3Shape, true);
    std::vector<sd::LongType> xgShape = {numTokens, numExperts, hiddenDim};
    NDArray xg('c', xgShape, input->dataType(), block.launchContext());
    gate3->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), input3, &xg, true);

    // output[t, d] = sum_{e,h} xg[t, e, h] * expertWeights[e, h, d]
    sd::ops::tensormmul contraction;
    auto status = contraction.execute({&xg, expertWeights}, {output}, {}, {2, 1, 2, 2, 0, 1}, {});

    delete routing3;
    delete gateFull;
    delete gate3;
    delete input3;

    REQUIRE_TRUE(status == Status::OK, 0, "moe_expert_ffn: expert contraction failed");
    return status;
}

DECLARE_TYPES(moe_expert_ffn) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_INTS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(moe_expert_ffn) {
    auto inShape = inputShape->at(0);
    auto expertShape = inputShape->at(1);
    std::vector<sd::LongType> outShape = {shape::sizeAt(inShape, static_cast<sd::LongType>(0)),
                                          shape::sizeAt(expertShape, static_cast<sd::LongType>(2))};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inShape), 'c', outShape));
}
#endif

}  // namespace ops
}  // namespace sd
