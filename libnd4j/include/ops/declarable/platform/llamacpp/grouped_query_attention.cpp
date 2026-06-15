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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "llamacppUtils.h"

#if HAVE_LLAMACPP

namespace sd {
namespace ops {
namespace platforms {

/**
 * Grouped Query Attention (GQA) using GGML
 *
 * GQA is a variant of multi-head attention where multiple query heads
 * share a single key-value head. This reduces memory usage and computation
 * while maintaining most of the performance of full multi-head attention.
 *
 * Used in models like LLaMA 2, Mistral, etc.
 *
 * Inputs:
 *   0: Q (query)  [batch, num_heads, seq_len, head_dim]
 *   1: K (key)    [batch, num_kv_heads, kv_seq_len, head_dim]
 *   2: V (value)  [batch, num_kv_heads, kv_seq_len, head_dim]
 *   3: mask (optional)
 *
 * Integer Arguments:
 *   0: num_heads (query heads)
 *   1: num_kv_heads (key-value heads, num_heads must be divisible by this)
 *
 * Output:
 *   0: attention output [batch, num_heads, seq_len, head_dim]
 */
static void groupedQueryAttentionLlamaCpp(NDArray* query, NDArray* key, NDArray* value,
                                           NDArray* mask, NDArray* output,
                                           int numHeads, int numKvHeads, float scale) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, key, "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, value, "value");

    // Calculate the number of query heads per KV head
    int groupSize = numHeads / numKvHeads;

    // For GQA, we need to expand K and V to match the number of query heads
    // This is done by repeating each KV head `groupSize` times
    if (groupSize > 1) {
        // Repeat K and V along the head dimension
        ggml_k = ggml_repeat(ctx, ggml_k, ggml_q);
        ggml_v = ggml_repeat(ctx, ggml_v, ggml_q);
    }

    // Apply flash attention with the expanded K and V
    struct ggml_tensor* ggml_output = ggml_flash_attn_ext(
        ctx, ggml_q, ggml_k, ggml_v,
        mask != nullptr ? llamacppUtils::createGgmlTensor(ctx, mask, "mask") : nullptr,
        scale,
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );

    ggml_set_name(ggml_output, "gqa_output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(grouped_query_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    NDArray* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) {
        return sd::Status::OK;
    }

    // Get number of heads
    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(1);
    int numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : key->sizeAt(1);

    // Validate head configuration
    REQUIRE_TRUE(numHeads % numKvHeads == 0, 0,
                 "GROUPED_QUERY_ATTENTION: num_heads (%d) must be divisible by num_kv_heads (%d)",
                 numHeads, numKvHeads);

    // Scale factor
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) :
        1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    groupedQueryAttentionLlamaCpp(query, key, value, mask, output, numHeads, numKvHeads, scale);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(grouped_query_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GROUPED_QUERY_ATTENTION OP");

    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    // Check supported data types
    req.expectTrue(
        makeInfoVariable(
            [query, key, value, output] {
                return llamacppUtils::isSupportedType(query->dataType()) &&
                       llamacppUtils::isSupportedType(key->dataType()) &&
                       llamacppUtils::isSupportedType(value->dataType()) &&
                       llamacppUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    // GQA requires 4D tensors
    req.expectEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectEq(makeInfoVariable(key->rankOf(), RANK_MSG_INPUT1), 4);
    req.expectEq(makeInfoVariable(value->rankOf(), RANK_MSG_INPUT2), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
