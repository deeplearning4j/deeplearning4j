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
// KV Cache operations for efficient LLM inference
// Supports paged attention, rolling buffer, and GQA/MQA cache patterns
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "llamacppUtils.h"

#if HAVE_LLAMACPP

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// KV_CACHE_UPDATE - Update KV cache with new key-value pairs
static void kvCacheUpdateLlamaCpp(NDArray* keyCache, NDArray* valueCache,
                                   NDArray* newKeys, NDArray* newValues,
                                   int seqPosition) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_k_cache = llamacppUtils::createGgmlTensor(ctx, keyCache, "k_cache");
    struct ggml_tensor* ggml_v_cache = llamacppUtils::createGgmlTensor(ctx, valueCache, "v_cache");
    struct ggml_tensor* ggml_new_k = llamacppUtils::createGgmlTensor(ctx, newKeys, "new_k");
    struct ggml_tensor* ggml_new_v = llamacppUtils::createGgmlTensor(ctx, newValues, "new_v");

    // Calculate offset for inserting new KV pairs
    size_t offset = seqPosition * newKeys->sizeAt(-1) * sizeof(float);

    // Update key cache
    struct ggml_tensor* ggml_k_updated = ggml_set(ctx, ggml_k_cache, ggml_new_k,
                                                   ggml_k_cache->nb[1], ggml_k_cache->nb[2],
                                                   ggml_k_cache->nb[3], offset);

    // Update value cache
    struct ggml_tensor* ggml_v_updated = ggml_set(ctx, ggml_v_cache, ggml_new_v,
                                                   ggml_v_cache->nb[1], ggml_v_cache->nb[2],
                                                   ggml_v_cache->nb[3], offset);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_k_updated);
    ggml_build_forward_expand(graph, ggml_v_updated);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_k_updated, keyCache);
    llamacppUtils::copyGgmlToNDArray(ggml_v_updated, valueCache);
}

PLATFORM_IMPL(kv_cache_update, ENGINE_CPU) {
    auto keyCache = INPUT_VARIABLE(0);
    auto valueCache = INPUT_VARIABLE(1);
    auto newKeys = INPUT_VARIABLE(2);
    auto newValues = INPUT_VARIABLE(3);

    if (keyCache->isEmpty() || newKeys->isEmpty()) return sd::Status::OK;

    int seqPosition = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    kvCacheUpdateLlamaCpp(keyCache, valueCache, newKeys, newValues, seqPosition);
    return sd::Status::OK;
}

PLATFORM_CHECK(kv_cache_update, ENGINE_CPU) {
    auto keyCache = INPUT_VARIABLE(0);
    auto valueCache = INPUT_VARIABLE(1);
    auto newKeys = INPUT_VARIABLE(2);

    Requirements req("LLAMACPP KV_CACHE_UPDATE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([keyCache, newKeys] {
        return llamacppUtils::isSupportedType(keyCache->dataType()) &&
               llamacppUtils::isSupportedType(newKeys->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// KV_CACHE_ATTENTION - Compute attention using KV cache
static void kvCacheAttentionLlamaCpp(NDArray* query, NDArray* keyCache, NDArray* valueCache,
                                      NDArray* output, int seqLen, float scale) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, keyCache, "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, valueCache, "v_cache");

    // Compute Q @ K^T
    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);

    // Scale
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);

    // Softmax
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);

    // Attention @ V
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(kv_cache_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    int seqLen = block.getIArguments()->size() > 0 ? INT_ARG(0) : keyCache->sizeAt(-2);
    float scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    kvCacheAttentionLlamaCpp(query, keyCache, valueCache, output, seqLen, scale);
    return sd::Status::OK;
}

PLATFORM_CHECK(kv_cache_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP KV_CACHE_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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
// PAGED_ATTENTION - Paged attention for memory-efficient KV cache
static void pagedAttentionLlamaCpp(NDArray* query, NDArray* keyCache, NDArray* valueCache,
                                    NDArray* blockTables, NDArray* contextLens,
                                    NDArray* output, int blockSize) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, keyCache, "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, valueCache, "v_cache");
    struct ggml_tensor* ggml_tables = llamacppUtils::createGgmlTensor(ctx, blockTables, "block_tables");

    // Simplified paged attention - gather blocks and compute attention
    // Real implementation would use specialized kernels
    struct ggml_tensor* ggml_k_gathered = ggml_get_rows(ctx, ggml_k, ggml_tables);
    struct ggml_tensor* ggml_v_gathered = ggml_get_rows(ctx, ggml_v, ggml_tables);

    // Compute attention with gathered KV
    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k_gathered);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    float scale = 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v_gathered, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(paged_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto blockTables = INPUT_VARIABLE(3);
    auto contextLens = INPUT_VARIABLE(4);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    int blockSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 16;

    pagedAttentionLlamaCpp(query, keyCache, valueCache, blockTables, contextLens, output, blockSize);
    return sd::Status::OK;
}

PLATFORM_CHECK(paged_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP PAGED_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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
// SLIDING_WINDOW_ATTENTION - Attention with sliding window mask
static void slidingWindowAttentionLlamaCpp(NDArray* query, NDArray* key, NDArray* value,
                                            NDArray* output, int windowSize) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, key, "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, value, "value");

    // Compute attention scores
    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);

    // Apply sliding window mask (positions outside window get -inf)
    // GGML has ggml_diag_mask_inf for causal masking
    struct ggml_tensor* ggml_masked = ggml_diag_mask_inf(ctx, ggml_scores, windowSize);

    // Scale and softmax
    float scale = 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));
    ggml_masked = ggml_scale(ctx, ggml_masked, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_masked);

    // Attention @ V
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(sliding_window_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty()) return sd::Status::OK;

    int windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 256;

    slidingWindowAttentionLlamaCpp(query, key, value, output, windowSize);
    return sd::Status::OK;
}

PLATFORM_CHECK(sliding_window_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SLIDING_WINDOW_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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
// GQA_ATTENTION - Grouped Query Attention with KV cache
static void gqaAttentionLlamaCpp(NDArray* query, NDArray* keyCache, NDArray* valueCache,
                                  NDArray* output, int numKvHeads) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, keyCache, "k_cache");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, valueCache, "v_cache");

    // For GQA, K and V have fewer heads than Q
    // Need to repeat K and V heads to match Q
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

    // Standard attention computation
    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k_repeated);
    struct ggml_tensor* ggml_scores = ggml_mul_mat(ctx, ggml_k_t, ggml_q);
    float scale = 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));
    ggml_scores = ggml_scale(ctx, ggml_scores, scale);
    struct ggml_tensor* ggml_attn = ggml_soft_max(ctx, ggml_scores);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v_repeated, ggml_attn);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(gqa_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto valueCache = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || keyCache->isEmpty()) return sd::Status::OK;

    int numKvHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : keyCache->sizeAt(1);

    gqaAttentionLlamaCpp(query, keyCache, valueCache, output, numKvHeads);
    return sd::Status::OK;
}

PLATFORM_CHECK(gqa_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto keyCache = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GQA_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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

#endif  // HAVE_LLAMACPP
