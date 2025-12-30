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
 * Flash Attention using GGML
 *
 * Flash Attention computes attention more efficiently by:
 * 1. Tiling the attention computation
 * 2. Avoiding materialization of the full attention matrix
 * 3. Using online softmax for numerical stability
 *
 * Inputs:
 *   0: Q (query) [batch, heads, seq_len, head_dim]
 *   1: K (key)   [batch, heads, kv_seq_len, head_dim]
 *   2: V (value) [batch, heads, kv_seq_len, head_dim]
 *   3: mask (optional) [batch, 1, seq_len, kv_seq_len]
 *
 * Output:
 *   0: attention output [batch, heads, seq_len, head_dim]
 */
static void flashAttentionLlamaCpp(NDArray* query, NDArray* key, NDArray* value,
                                    NDArray* mask, NDArray* output, float scale) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);  // 128MB workspace

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, key, "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, value, "value");

    // Use GGML's flash attention implementation
    struct ggml_tensor* ggml_output = ggml_flash_attn_ext(
        ctx, ggml_q, ggml_k, ggml_v,
        mask != nullptr ? llamacppUtils::createGgmlTensor(ctx, mask, "mask") : nullptr,
        scale,
        0.0f,  // max_bias (for ALiBi, set to 0 for standard attention)
        0.0f   // logit_softcap (no capping)
    );

    ggml_set_name(ggml_output, "flash_attn_output");

    // Build and execute computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(flash_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    NDArray* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) {
        return sd::Status::OK;
    }

    // Scale factor (default: 1/sqrt(head_dim))
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) :
        1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    flashAttentionLlamaCpp(query, key, value, mask, output, scale);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(flash_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP FLASH_ATTENTION OP");

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

    // Flash attention requires 4D tensors: [batch, heads, seq_len, head_dim]
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
