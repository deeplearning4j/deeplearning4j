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
// llama.cpp-compat normalization / MoE / attention ops implemented natively by
// composition. Where the original ggml/llamacpp impl was nonstandard (noted
// inline), the native version follows the correct/standard definition.
//

#include <system/op_boilerplate.h>

#include <helpers/ConstantShapeHelper.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/headers/nn.h>

#include <vector>

namespace sd {
namespace ops {

// ─── group_norm: normalize over (C/G channels + spatial) per (N, group) ──────
#if NOT_EXCLUDED(OP_group_norm)
CUSTOM_OP_IMPL(group_norm, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);   // [N, C, *spatial]
    auto output = OUTPUT_VARIABLE(0);

    const int rank = input->rankOf();
    REQUIRE_TRUE(rank >= 2 && rank <= 4, 0, "group_norm: input rank must be in [2,4], got %i", rank);
    const LongType numGroups = block.getIArguments()->size() > 0 ? INT_ARG(0) : 32;
    const double eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5;

    const LongType N = input->sizeAt(0), C = input->sizeAt(1);
    REQUIRE_TRUE(numGroups > 0 && C % numGroups == 0, 0,
                 "group_norm: channels (%lld) must be divisible by numGroups (%lld)",
                 (long long)C, (long long)numGroups);
    if (input->isEmpty()) return Status::OK;

    // reshape [N, C, *spatial] → [N, G, C/G, *spatial]; reduce over axes >= 2
    std::vector<LongType> groupedShape;
    groupedShape.push_back(N);
    groupedShape.push_back(numGroups);
    groupedShape.push_back(C / numGroups);
    for (int i = 2; i < rank; i++) groupedShape.push_back(input->sizeAt(i));
    const int gRank = static_cast<int>(groupedShape.size());

    std::vector<LongType> reduceAxes;
    std::vector<LongType> reducedShape = groupedShape;
    for (int i = 2; i < gRank; i++) { reduceAxes.push_back(i); reducedShape[i] = 1; }

    NDArray* grouped = input->reshape('c', groupedShape);

    NDArray mean('c', reducedShape, input->dataType(), block.launchContext());
    grouped->reduceAlongDimension(reduce::Mean, &mean, &reduceAxes, true, false);

    NDArray centered('c', groupedShape, input->dataType(), block.launchContext());
    grouped->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &mean, &centered, true);

    NDArray sq('c', groupedShape, input->dataType(), block.launchContext());
    centered.applyPairwiseTransform(pairwise::Multiply, &centered, &sq, nullptr);

    NDArray var('c', reducedShape, input->dataType(), block.launchContext());
    sq.reduceAlongDimension(reduce::Mean, &var, &reduceAxes, true, false);  // population variance
    var.applyScalar(scalar::Add, eps, &var);
    var.applyTransform(transform::Sqrt, &var);

    NDArray normalized('c', groupedShape, input->dataType(), block.launchContext());
    centered.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &var, &normalized, true);

    std::vector<LongType> origShape(rank);
    for (int i = 0; i < rank; i++) origShape[i] = input->sizeAt(i);
    NDArray* normFlat = normalized.reshape('c', origShape);
    output->assign(normFlat);

    delete grouped;
    delete normFlat;
    return Status::OK;
}
DECLARE_TYPES(group_norm) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(group_norm) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

// ─── l2_normalize: x / sqrt(sum(x^2) + eps) along the last dim ───────────────
#if NOT_EXCLUDED(OP_l2_normalize)
CUSTOM_OP_IMPL(l2_normalize, 1, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const int rank = input->rankOf();
    REQUIRE_TRUE(rank >= 1, 0, "l2_normalize: input rank must be >= 1");
    const double eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-12;
    if (input->isEmpty()) return Status::OK;

    std::vector<LongType> lastAxis = {rank - 1};
    std::vector<LongType> fullShape(rank), reducedShape(rank);
    for (int i = 0; i < rank; i++) fullShape[i] = reducedShape[i] = input->sizeAt(i);
    reducedShape[rank - 1] = 1;

    NDArray sq('c', fullShape, input->dataType(), block.launchContext());
    input->applyPairwiseTransform(pairwise::Multiply, input, &sq, nullptr);

    NDArray norm('c', reducedShape, input->dataType(), block.launchContext());
    sq.reduceAlongDimension(reduce::Sum, &norm, &lastAxis, true, false);
    norm.applyScalar(scalar::Add, eps, &norm);
    norm.applyTransform(transform::Sqrt, &norm);

    input->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &norm, output, true);
    return Status::OK;
}
DECLARE_TYPES(l2_normalize) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(l2_normalize) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

// ─── load_balance_loss: Switch-Transformer aux loss ─────────────────────────
// loss = E * sum_e( mean_batch(probs[:,e]) * mean_batch(mask[:,e]) ).
// NOTE: corrects the llamacpp/ggml impl, which reduced over the expert axis
// (ggml_mean → per-row) and dropped the numExperts factor.
#if NOT_EXCLUDED(OP_load_balance_loss)
CUSTOM_OP_IMPL(load_balance_loss, 2, 1, false, 0, 0) {
    auto routingProbs = INPUT_VARIABLE(0);  // [batch, E]
    auto expertMask = INPUT_VARIABLE(1);    // [batch, E] (one-hot / selection mask)
    auto output = OUTPUT_VARIABLE(0);       // [1]

    REQUIRE_TRUE(routingProbs->rankOf() == 2 && expertMask->isSameShape(routingProbs), 0,
                 "load_balance_loss: both inputs must be rank 2 [batch, numExperts] and same-shaped");
    if (routingProbs->isEmpty()) { output->p(0, 0.0); return Status::OK; }

    const LongType numExperts = routingProbs->sizeAt(1);
    std::vector<LongType> axis0 = {0};
    NDArray* meanProbs = routingProbs->reduceAlongDimension(reduce::Mean, &axis0, false);  // [E]
    NDArray* meanMask = expertMask->reduceAlongDimension(reduce::Mean, &axis0, false);     // [E]

    *meanProbs *= *meanMask;
    NDArray* s = meanProbs->reduceNumber(reduce::Sum);
    output->p(0, static_cast<double>(numExperts) * s->e<double>(0));

    delete meanProbs;
    delete meanMask;
    delete s;
    return Status::OK;
}
DECLARE_TYPES(load_balance_loss) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(load_balance_loss) {
    std::vector<sd::LongType> outShape = {1};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inputShape->at(0)), 'c', outShape));
}
#endif

// ─── sparse_mul_mat: per-token expert matmul ────────────────────────────────
// out[t, d] = sum_h input[t, h] * sparseWeights[indices[t], h, d]
#if NOT_EXCLUDED(OP_sparse_mul_mat)
CUSTOM_OP_IMPL(sparse_mul_mat, 3, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);          // [T, H]
    auto sparseWeights = INPUT_VARIABLE(1);  // [E, H, D]
    auto indices = INPUT_VARIABLE(2);        // [T] int
    auto output = OUTPUT_VARIABLE(0);        // [T, D]

    REQUIRE_TRUE(input->rankOf() == 2, 0, "sparse_mul_mat: input must be rank 2 [T,H], got %i", input->rankOf());
    REQUIRE_TRUE(sparseWeights->rankOf() == 3, 0,
                 "sparse_mul_mat: sparseWeights must be rank 3 [E,H,D], got %i", sparseWeights->rankOf());
    REQUIRE_TRUE(sparseWeights->sizeAt(1) == input->sizeAt(1), 0,
                 "sparse_mul_mat: weights H (%lld) must match input H (%lld)",
                 (long long)sparseWeights->sizeAt(1), (long long)input->sizeAt(1));
    if (input->isEmpty()) { output->nullify(); return Status::OK; }

    const LongType T = input->sizeAt(0), H = input->sizeAt(1), D = sparseWeights->sizeAt(2);

    // gather the per-token weight matrices: [T, H, D]
    sd::ops::gather gatherOp;
    auto gathered = gatherOp.evaluate({sparseWeights, indices}, {0});  // iArg 0 = axis
    REQUIRE_TRUE(gathered.status() == Status::OK, 0, "sparse_mul_mat: gather failed");
    NDArray* g = gathered.at(0);  // [T, H, D]

    // out[t,d] = sum_h input[t,h] * g[t,h,d]
    std::vector<LongType> in3Shape = {T, H, 1};
    NDArray* in3 = input->reshape('c', in3Shape);
    std::vector<LongType> prodShape = {T, H, D};
    NDArray prod('c', prodShape, input->dataType(), block.launchContext());
    in3->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), g, &prod, true);

    std::vector<LongType> hAxis = {1};
    prod.reduceAlongDimension(reduce::Sum, output, &hAxis, false, false);

    delete in3;
    return Status::OK;
}
DECLARE_TYPES(sparse_mul_mat) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {ALL_INTS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(sparse_mul_mat) {
    auto inShape = inputShape->at(0);
    auto wShape = inputShape->at(1);
    std::vector<sd::LongType> outShape = {shape::sizeAt(inShape, static_cast<sd::LongType>(0)),
                                          shape::sizeAt(wShape, static_cast<sd::LongType>(2))};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inShape), 'c', outShape));
}
#endif

// ─── kv_cache_attention: non-causal SDPA over a KV cache ────────────────────
// Thin adapter over grouped_query_attention (full attention; scale forwarded,
// seqLen iarg ignored — derivable from the K cache shape).
#if NOT_EXCLUDED(OP_kv_cache_attention)
CUSTOM_OP_IMPL(kv_cache_attention, 3, 1, false, -2, -2) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(query->rankOf() == 4, 0, "kv_cache_attention: query must be rank 4 [B,S,H,Dh], got %i",
                 query->rankOf());
    const double scaleVal = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;  // 0 = auto (1/sqrt(Dh))

    sd::ops::grouped_query_attention gqa;
    auto status = gqa.execute({query, keyCache, valueCache}, {output}, {scaleVal}, {}, {false});  // non-causal
    REQUIRE_TRUE(status == Status::OK, 0, "kv_cache_attention: delegation to grouped_query_attention failed");
    return status;
}
DECLARE_TYPES(kv_cache_attention) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(kv_cache_attention) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

// embedding_lookup_bp is the same scatter-add gradient as get_rows_bp
// (implemented in llm_compat_ops.cpp).
#if NOT_EXCLUDED(OP_embedding_lookup_bp)
DECLARE_SYN(embedding_lookup_bp, get_rows_bp);
#endif

}  // namespace ops
}  // namespace sd
