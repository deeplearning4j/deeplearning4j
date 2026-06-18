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
// CUDA implementations of MoE and KV cache operations using GGML kernels
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <math/templatemath.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// MOE_GATE - CUDA
PLATFORM_IMPL(moe_gate, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gateWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || gateWeights->isEmpty()) return sd::Status::OK;

    int topK = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_gate = llamacppUtils::createGgmlTensorCuda(ctx, gateWeights, ctx.getBackend(), "gate");

    struct ggml_tensor* ggml_logits = ggml_mul_mat(ctx, ggml_gate, ggml_input);
    struct ggml_tensor* ggml_probs = ggml_soft_max(ctx, ggml_logits);
    struct ggml_tensor* ggml_output = ggml_top_k(ctx, ggml_probs, topK);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(moe_gate, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gateWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MOE_GATE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, gateWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gateWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MOE_EXPERT_FFN - CUDA
PLATFORM_IMPL(moe_expert_ffn, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto expertWeights = INPUT_VARIABLE(1);
    auto routingWeights = INPUT_VARIABLE(2);
    auto expertIndices = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_experts = llamacppUtils::createGgmlTensorCuda(ctx, expertWeights, ctx.getBackend(), "experts");
    struct ggml_tensor* ggml_routing = llamacppUtils::createGgmlTensorCuda(ctx, routingWeights, ctx.getBackend(), "routing");

    struct ggml_tensor* ggml_ffn = ggml_mul_mat(ctx, ggml_experts, ggml_input);
    struct ggml_tensor* ggml_output = ggml_mul(ctx, ggml_ffn, ggml_routing);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(moe_expert_ffn, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto expertWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MOE_EXPERT_FFN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, expertWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(expertWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SPARSE_MUL_MAT - CUDA
PLATFORM_IMPL(sparse_mul_mat, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto sparseWeights = INPUT_VARIABLE(1);
    auto indices = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || sparseWeights->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensorCuda(ctx, sparseWeights, ctx.getBackend(), "weights");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensorCuda(ctx, indices, ctx.getBackend(), "indices");

    struct ggml_tensor* ggml_selected = ggml_get_rows(ctx, ggml_weights, ggml_indices);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_selected, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(sparse_mul_mat, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto sparseWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SPARSE_MUL_MAT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, sparseWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(sparseWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// LOAD_BALANCE_LOSS - CUDA
PLATFORM_IMPL(load_balance_loss, ENGINE_CUDA) {
    auto routingProbs = INPUT_VARIABLE(0);
    auto expertMask = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (routingProbs->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_probs = llamacppUtils::createGgmlTensorCuda(ctx, routingProbs, ctx.getBackend(), "probs");
    struct ggml_tensor* ggml_mask = llamacppUtils::createGgmlTensorCuda(ctx, expertMask, ctx.getBackend(), "mask");

    struct ggml_tensor* ggml_mean_probs = ggml_mean(ctx, ggml_probs);
    struct ggml_tensor* ggml_mean_mask = ggml_mean(ctx, ggml_mask);
    struct ggml_tensor* ggml_product = ggml_mul(ctx, ggml_mean_probs, ggml_mean_mask);
    struct ggml_tensor* ggml_output = ggml_sum(ctx, ggml_product);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(load_balance_loss, ENGINE_CUDA) {
    auto routingProbs = INPUT_VARIABLE(0);
    auto expertMask = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA LOAD_BALANCE_LOSS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([routingProbs, expertMask, output] {
        return llamacppUtils::isSupportedType(routingProbs->dataType()) &&
               llamacppUtils::isSupportedType(expertMask->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(routingProbs->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// KV_CACHE_UPDATE - CUDA
PLATFORM_IMPL(kv_cache_update, ENGINE_CUDA) {
    auto keyCache = INPUT_VARIABLE(0);
    auto valueCache = INPUT_VARIABLE(1);
    auto newKeys = INPUT_VARIABLE(2);
    auto newValues = INPUT_VARIABLE(3);

    if (keyCache->isEmpty() || newKeys->isEmpty()) return sd::Status::OK;

    int seqPosition = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_k_cache = llamacppUtils::createGgmlTensorCuda(ctx, keyCache, ctx.getBackend(), "k_cache");
    struct ggml_tensor* ggml_v_cache = llamacppUtils::createGgmlTensorCuda(ctx, valueCache, ctx.getBackend(), "v_cache");
    struct ggml_tensor* ggml_new_k = llamacppUtils::createGgmlTensorCuda(ctx, newKeys, ctx.getBackend(), "new_k");
    struct ggml_tensor* ggml_new_v = llamacppUtils::createGgmlTensorCuda(ctx, newValues, ctx.getBackend(), "new_v");

    size_t offset = seqPosition * newKeys->sizeAt(-1) * sizeof(float);

    struct ggml_tensor* ggml_k_updated = ggml_set(ctx, ggml_k_cache, ggml_new_k,
                                                   ggml_k_cache->nb[1], ggml_k_cache->nb[2],
                                                   ggml_k_cache->nb[3], offset);
    struct ggml_tensor* ggml_v_updated = ggml_set(ctx, ggml_v_cache, ggml_new_v,
                                                   ggml_v_cache->nb[1], ggml_v_cache->nb[2],
                                                   ggml_v_cache->nb[3], offset);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_k_updated);
    ggml_build_forward_expand(graph, ggml_v_updated);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_k_updated, keyCache, ctx.getBackend());
    llamacppUtils::copyGgmlCudaToNDArray(ggml_v_updated, valueCache, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(kv_cache_update, ENGINE_CUDA) {
    auto keyCache = INPUT_VARIABLE(0);
    auto newKeys = INPUT_VARIABLE(2);

    Requirements req("LLAMACPP CUDA KV_CACHE_UPDATE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([keyCache, newKeys] {
        return llamacppUtils::isSupportedType(keyCache->dataType()) &&
               llamacppUtils::isSupportedType(newKeys->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// KV_CACHE_ATTENTION - CUDA
PLATFORM_IMPL(kv_cache_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    float scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, keyCache, ctx.getBackend(), "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, valueCache, ctx.getBackend(), "v_cache");

    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(kv_cache_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA KV_CACHE_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, keyCache, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(keyCache->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// PAGED_ATTENTION - CUDA
PLATFORM_IMPL(paged_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto blockTables = INPUT_VARIABLE(3);
    auto contextLens = INPUT_VARIABLE(4);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, keyCache, ctx.getBackend(), "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, valueCache, ctx.getBackend(), "v_cache");
    struct ggml_tensor* ggml_tables = llamacppUtils::createGgmlTensorCuda(ctx, blockTables, ctx.getBackend(), "block_tables");

    struct ggml_tensor* ggml_k_gathered = ggml_get_rows(ctx, ggml_k, ggml_tables);
    struct ggml_tensor* ggml_v_gathered = ggml_get_rows(ctx, ggml_v, ggml_tables);

    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k_gathered);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    float scale = 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v_gathered, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(paged_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA PAGED_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, keyCache, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(keyCache->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 5);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SLIDING_WINDOW_ATTENTION - CUDA
PLATFORM_IMPL(sliding_window_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty()) return sd::Status::OK;

    int windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 256;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, key, ctx.getBackend(), "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, value, ctx.getBackend(), "value");

    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    struct ggml_tensor* ggml_masked = ggml_diag_mask_inf(ctx, ggml_scores, windowSize);
    float scale = 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));
    ggml_masked = ggml_scale(ctx, ggml_masked, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_masked);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(sliding_window_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SLIDING_WINDOW_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, key, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(key->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GQA_ATTENTION - CUDA
PLATFORM_IMPL(gqa_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    int numKvHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : keyCache->sizeAt(1);

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, keyCache, ctx.getBackend(), "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, valueCache, ctx.getBackend(), "v_cache");

    int numQHeads = query->sizeAt(1);
    int repeatFactor = numQHeads / numKvHeads;

    struct ggml_tensor* ggml_k_repeated = ggml_repeat(ctx, ggml_k,
        ggml_new_tensor_4d(ctx, ggml_k->type,
                           ggml_k->ne[0], ggml_k->ne[1] * repeatFactor,
                           ggml_k->ne[2], ggml_k->ne[3]));
    struct ggml_tensor* ggml_v_repeated = ggml_repeat(ctx, ggml_v,
        ggml_new_tensor_4d(ctx, ggml_v->type,
                           ggml_v->ne[0], ggml_v->ne[1] * repeatFactor,
                           ggml_v->ne[2], ggml_v->ne[3]));

    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k_repeated);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    float scale = 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v_repeated, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(gqa_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GQA_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, keyCache, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(keyCache->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 3);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
