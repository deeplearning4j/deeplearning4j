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
// Gemma 4 operations using GGML kernels (CPU platform implementations):
// per_layer_embedding, shared_kv_attention, dual_rope
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
// PER_LAYER_EMBEDDING - Per-layer embedding residual addition (CPU)
// output = hidden_states + ple_weight[token_ids] * scale
PLATFORM_IMPL(per_layer_embedding, ENGINE_CPU) {
    auto hiddenStates = INPUT_VARIABLE(0);   // [B, S, D]
    auto pleWeight = INPUT_VARIABLE(1);      // [vocab, D]
    auto tokenIds = INPUT_VARIABLE(2);       // [B, S]
    auto output = OUTPUT_VARIABLE(0);        // [B, S, D]

    if (hiddenStates->isEmpty()) return sd::Status::OK;

    double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    // GGML: embedding lookup + scale + add
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_hidden = llamacppUtils::createGgmlTensor(ctx, hiddenStates, "hidden");
    struct ggml_tensor* ggml_weight = llamacppUtils::createGgmlTensor(ctx, pleWeight, "ple_weight");
    struct ggml_tensor* ggml_ids = llamacppUtils::createGgmlTensor(ctx, tokenIds, "token_ids");

    // get_rows performs the embedding lookup: ple_weight[token_ids] -> [B, S, D]
    struct ggml_tensor* lookup = ggml_get_rows(ctx, ggml_weight, ggml_ids);
    ggml_set_name(lookup, "lookup");

    // Scale the lookup if needed
    struct ggml_tensor* scaled = lookup;
    if (scale != 1.0) {
        scaled = ggml_scale(ctx, lookup, static_cast<float>(scale));
        ggml_set_name(scaled, "scaled");
    }

    // Add to hidden states
    struct ggml_tensor* result = ggml_add(ctx, ggml_hidden, scaled);
    ggml_set_name(result, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(result, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(per_layer_embedding, ENGINE_CPU) {
    auto hiddenStates = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP PER_LAYER_EMBEDDING OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([hiddenStates, output] {
        return llamacppUtils::isSupportedType(hiddenStates->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(hiddenStates->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SHARED_KV_ATTENTION - Grouped-query attention with shared K/V from donor layer (CPU)
PLATFORM_IMPL(shared_kv_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);         // [B, S, H, D]
    auto sharedKey = INPUT_VARIABLE(1);     // [B, kvS, kvH, D]
    auto sharedValue = INPUT_VARIABLE(2);   // [B, kvS, kvH, D]
    auto output = OUTPUT_VARIABLE(0);       // [B, S, H, D]

    if (query->isEmpty()) return sd::Status::OK;

    NDArray* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : query->sizeAt(2);
    int numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : sharedKey->sizeAt(2);

    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "q");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, sharedKey, "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, sharedValue, "v");

    struct ggml_tensor* ggml_mask = mask != nullptr ?
        llamacppUtils::createGgmlTensor(ctx, mask, "mask") : nullptr;

    // Use GGML flash attention if available, otherwise standard attention
    struct ggml_tensor* attn_out;
    float scale_val = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 0.0f;
    if (scale_val == 0.0f) {
        scale_val = 1.0f / sqrtf(static_cast<float>(query->sizeAt(3)));
    }

    // GGML flash_attn_ext handles GQA natively (KV head broadcasting)
    attn_out = ggml_flash_attn_ext(ctx, ggml_q, ggml_k, ggml_v, ggml_mask, scale_val, 0.0f, 0.0f);
    ggml_set_name(attn_out, "attn_output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, attn_out);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(attn_out, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(shared_kv_attention, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto sharedKey = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SHARED_KV_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([query, sharedKey, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(sharedKey->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectEq(makeInfoVariable(sharedKey->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DUAL_ROPE - Dual Rotary Position Embedding (CPU)
// Standard RoPE for sliding-window layers, proportional RoPE for global layers
PLATFORM_IMPL(dual_rope, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);     // [B, S, H, D]
    auto output = OUTPUT_VARIABLE(0);   // [B, S, H, D]

    if (input->isEmpty()) return sd::Status::OK;

    int attentionType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int positionOffset = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    float localFreqBase = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 10000.0f;
    float globalFreqBase = block.getTArguments()->size() > 1 ? static_cast<float>(T_ARG(1)) : 1000000.0f;

    // Select freq_base based on attention type
    float freqBase = (attentionType == 0) ? localFreqBase : globalFreqBase;

    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // GGML rope with the appropriate frequency base
    // ggml_rope_ext applies RoPE with configurable freq_base
    int n_dims = input->sizeAt(3);
    int mode = 0;  // standard RoPE
    struct ggml_tensor* rope_out = ggml_rope_ext(ctx, ggml_input,
                                                  nullptr,  // position tensor (auto from shape)
                                                  nullptr,  // freq factors
                                                  n_dims, mode,
                                                  0,  // n_ctx_orig
                                                  freqBase, 1.0f,  // freq_base, freq_scale
                                                  0.0f, 0.0f, 0.0f, 0.0f);  // ext factors
    ggml_set_name(rope_out, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rope_out);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(rope_out, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(dual_rope, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP DUAL_ROPE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
