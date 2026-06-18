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
// CUDA implementations of Gemma 4 operations using GGML kernels:
// per_layer_embedding, shared_kv_attention, dual_rope
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// PER_LAYER_EMBEDDING - Per-layer embedding residual addition (CUDA)
PLATFORM_IMPL(per_layer_embedding, ENGINE_CUDA) {
    auto hiddenStates = INPUT_VARIABLE(0);
    auto pleWeight = INPUT_VARIABLE(1);
    auto tokenIds = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (hiddenStates->isEmpty()) return sd::Status::OK;

    double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_hidden = llamacppUtils::createGgmlTensorCuda(ctx, hiddenStates, ctx.getBackend(), "hidden");
    struct ggml_tensor* ggml_weight = llamacppUtils::createGgmlTensorCuda(ctx, pleWeight, ctx.getBackend(), "ple_weight");
    struct ggml_tensor* ggml_ids = llamacppUtils::createGgmlTensorCuda(ctx, tokenIds, ctx.getBackend(), "token_ids");

    struct ggml_tensor* lookup = ggml_get_rows(ctx, ggml_weight, ggml_ids);
    ggml_set_name(lookup, "lookup");

    struct ggml_tensor* scaled = lookup;
    if (scale != 1.0) {
        scaled = ggml_scale(ctx, lookup, static_cast<float>(scale));
        ggml_set_name(scaled, "scaled");
    }

    struct ggml_tensor* result = ggml_add(ctx, ggml_hidden, scaled);
    ggml_set_name(result, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(result, output, ctx.getBackend());

    return sd::Status::OK;
}

PLATFORM_CHECK(per_layer_embedding, ENGINE_CUDA) {
    auto hiddenStates = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA PER_LAYER_EMBEDDING OP");
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
// SHARED_KV_ATTENTION - Grouped-query attention with shared K/V (CUDA)
PLATFORM_IMPL(shared_kv_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto sharedKey = INPUT_VARIABLE(1);
    auto sharedValue = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty()) return sd::Status::OK;

    NDArray* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    float scale_val = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 0.0f;
    if (scale_val == 0.0f) {
        scale_val = 1.0f / sqrtf(static_cast<float>(query->sizeAt(3)));
    }

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "q");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, sharedKey, ctx.getBackend(), "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, sharedValue, ctx.getBackend(), "v");

    struct ggml_tensor* ggml_mask = mask != nullptr ?
        llamacppUtils::createGgmlTensorCuda(ctx, mask, ctx.getBackend(), "mask") : nullptr;

    struct ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, ggml_q, ggml_k, ggml_v, ggml_mask, scale_val, 0.0f, 0.0f);
    ggml_set_name(attn_out, "attn_output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, attn_out);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(attn_out, output, ctx.getBackend());

    return sd::Status::OK;
}

PLATFORM_CHECK(shared_kv_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto sharedKey = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SHARED_KV_ATTENTION OP");
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
// DUAL_ROPE - Dual Rotary Position Embedding (CUDA)
PLATFORM_IMPL(dual_rope, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int attentionType = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    float localFreqBase = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 10000.0f;
    float globalFreqBase = block.getTArguments()->size() > 1 ? static_cast<float>(T_ARG(1)) : 1000000.0f;
    float freqBase = (attentionType == 0) ? localFreqBase : globalFreqBase;

    int n_dims = input->sizeAt(3);

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* rope_out = ggml_rope_ext(ctx, ggml_input,
                                                  nullptr, nullptr,
                                                  n_dims, 0,
                                                  0, freqBase, 1.0f,
                                                  0.0f, 0.0f, 0.0f, 0.0f);
    ggml_set_name(rope_out, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rope_out);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(rope_out, output, ctx.getBackend());

    return sd::Status::OK;
}

PLATFORM_CHECK(dual_rope, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA DUAL_ROPE OP");
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

#endif  // HAVE_LLAMACPP && defined(GGML_USE_CUDA)
