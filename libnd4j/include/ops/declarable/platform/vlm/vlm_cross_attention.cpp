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

#include "vlmUtils.h"

#if HAVE_VLM

namespace sd {
namespace ops {
namespace platforms {

/**
 * Cross attention operation for VLM using GGML
 *
 * Computes attention between query (language) and key/value (vision) features.
 * This enables the language model to attend to relevant visual features.
 *
 * Inputs:
 *   query: [B, seq_len_q, dim] - Language features
 *   key:   [B, seq_len_kv, dim] - Vision features (keys)
 *   value: [B, seq_len_kv, dim] - Vision features (values)
 *
 * Output: [B, seq_len_q, dim] - Attended features
 *
 * Args:
 *   INT_ARG(0): num_heads
 *   T_ARG(0): scale (optional, defaults to 1/sqrt(head_dim))
 */
static void crossAttentionVlm(NDArray* query, NDArray* key, NDArray* value,
                               NDArray* output, int numHeads, float scale) {
    vlmUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_q = vlmUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = vlmUtils::createGgmlTensor(ctx, key, "key");
    struct ggml_tensor* ggml_v = vlmUtils::createGgmlTensor(ctx, value, "value");

    // Compute Q @ K^T
    struct ggml_tensor* ggml_k_t = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* qk = ggml_mul_mat(ctx, ggml_k_t, ggml_q);

    // Scale
    struct ggml_tensor* qk_scaled = ggml_scale(ctx, qk, scale);

    // Softmax along last dimension
    struct ggml_tensor* attn_weights = ggml_soft_max(ctx, qk_scaled);

    // Attention output: softmax(QK^T / sqrt(d)) @ V
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_v, attn_weights);
    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    vlmUtils::executeGgmlGraph(ctx, graph);

    vlmUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(vlm_cross_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);     // [B, seq_len_q, dim]
    auto key = INPUT_VARIABLE(1);       // [B, seq_len_kv, dim]
    auto value = INPUT_VARIABLE(2);     // [B, seq_len_kv, dim]
    auto output = OUTPUT_VARIABLE(0);   // [B, seq_len_q, dim]

    if (query->isEmpty()) return sd::Status::OK;

    int numHeads = INT_ARG(0);
    int headDim = query->sizeAt(-1) / numHeads;
    float scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(headDim));

    crossAttentionVlm(query, key, value, output, numHeads, scale);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(vlm_cross_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("VLM CROSS_ATTENTION OP");

    req.expectTrue(block.isUseVLM(), IS_USE_VLM_MSG);

    req.expectTrue(
        makeInfoVariable(
            [query, key, value, output] {
                return vlmUtils::isSupportedType(query->dataType()) &&
                       vlmUtils::isSupportedType(key->dataType()) &&
                       vlmUtils::isSupportedType(value->dataType()) &&
                       vlmUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 3);
    req.expectEq(makeInfoVariable(key->rankOf(), RANK_MSG_INPUT1), 3);
    req.expectEq(makeInfoVariable(value->rankOf(), "RANK INPUT2"), 3);

    // Key and value must have same sequence length
    req.expectEq(makeInfoVariable(key->sizeAt(1), "KEY_SEQ_LEN"),
                 makeInfoVariable(value->sizeAt(1), "VALUE_SEQ_LEN"));

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_VLM
