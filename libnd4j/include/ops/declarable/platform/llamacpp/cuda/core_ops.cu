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
// Core LLM operations using GGML CUDA kernels:
// matmul, quantized_matmul, rms_norm, rope, silu, softmax, gelu, layer_norm,
// grouped_query_attention, flash_attention
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <math/templatemath.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

#include <ggml-cuda.h>
#include <ggml-backend.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// MATMUL - Matrix multiplication on CUDA
static void matmulCuda(NDArray* a, NDArray* b, NDArray* c) {
    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_a = llamacppUtils::createGgmlTensorCuda(ctx, a, ctx.getBackend(), "a");
    struct ggml_tensor* ggml_b = llamacppUtils::createGgmlTensorCuda(ctx, b, ctx.getBackend(), "b");

    struct ggml_tensor* ggml_c = ggml_mul_mat(ctx, ggml_a, ggml_b);
    ggml_set_name(ggml_c, "c");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_c);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_c, c, ctx.getBackend());
}

PLATFORM_IMPL(matmul, ENGINE_CUDA) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    if (a->isEmpty() || b->isEmpty()) return sd::Status::OK;

    matmulCuda(a, b, c);
    return sd::Status::OK;
}

PLATFORM_CHECK(matmul, ENGINE_CUDA) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MATMUL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([a, b, c] {
        return llamacppUtils::isSupportedType(a->dataType()) &&
               llamacppUtils::isSupportedType(b->dataType()) &&
               llamacppUtils::isSupportedType(c->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(a->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(b->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// RMS_NORM - RMS Normalization on CUDA
static void rmsNormCuda(NDArray* input, NDArray* weight, NDArray* output, float eps) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_rms_norm(ctx, ggml_input, eps);

    // Apply weight/gamma if provided
    if (weight != nullptr && !weight->isEmpty()) {
        struct ggml_tensor* ggml_weight = llamacppUtils::createGgmlTensorCuda(ctx, weight, ctx.getBackend(), "weight");
        ggml_output = ggml_mul(ctx, ggml_output, ggml_weight);
    }

    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(rms_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    NDArray* weight = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;
    rmsNormCuda(input, weight, output, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(rms_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA RMS_NORM OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ROPE - Rotary Position Embedding on CUDA
static void ropeCuda(NDArray* input, NDArray* output, int mode, int nPast, int nDims,
                     int nCtx, float freqBase, float freqScale) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_rope(ctx, ggml_input, nullptr, nDims, mode);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(rope, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int nPast = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int nDims = block.getIArguments()->size() > 2 ? INT_ARG(2) : input->sizeAt(-1);
    int nCtx = block.getIArguments()->size() > 3 ? INT_ARG(3) : 2048;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    ropeCuda(input, output, mode, nPast, nDims, nCtx, freqBase, freqScale);
    return sd::Status::OK;
}

PLATFORM_CHECK(rope, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ROPE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SILU - SiLU/Swish activation on CUDA
static void siluCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_silu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(silu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    siluCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(silu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SILU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SOFTMAX - Softmax on CUDA
static void softmaxCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_soft_max(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(softmax, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    softmaxCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SOFTMAX OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GELU - GELU activation on CUDA
static void geluCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_gelu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(gelu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    geluCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(gelu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GELU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// LAYER_NORM - Layer normalization on CUDA
static void layerNormCuda(NDArray* input, NDArray* output, float eps) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_norm(ctx, ggml_input, eps);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(layer_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;
    layerNormCuda(input, output, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA LAYER_NORM OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// QUANTIZED_MATMUL - Quantized matrix multiplication on CUDA
PLATFORM_IMPL(quantized_matmul, ENGINE_CUDA) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    if (a->isEmpty() || b->isEmpty()) return sd::Status::OK;

    // For now, use regular matmul - quantization handled by GGML internally
    matmulCuda(a, b, c);
    return sd::Status::OK;
}

PLATFORM_CHECK(quantized_matmul, ENGINE_CUDA) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA QUANTIZED_MATMUL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectLessEq(makeInfoVariable(a->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(b->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GROUPED_QUERY_ATTENTION - GQA on CUDA
static void gqaCuda(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                    int numHeads, int numKvHeads, float scale, bool isCausal) {
    llamacppUtils::GgmlCudaContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, key, ctx.getBackend(), "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, value, ctx.getBackend(), "value");

    // Compute Q * K^T
    struct ggml_tensor* kT = ggml_transpose(ctx, ggml_k);
    struct ggml_tensor* qk = ggml_mul_mat(ctx, ggml_q, kT);

    // Scale
    qk = ggml_scale(ctx, qk, scale);

    // Apply causal mask if needed
    if (isCausal) {
        qk = ggml_diag_mask_inf(ctx, qk, 0);
    }

    // Softmax
    struct ggml_tensor* attn = ggml_soft_max(ctx, qk);

    // Attention output
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, attn, ggml_v);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(grouped_query_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) return sd::Status::OK;

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : 8;
    int numKvHeads = block.getIArguments()->size() > 1 ? INT_ARG(1) : numHeads;
    bool isCausal = block.getIArguments()->size() > 2 ? INT_ARG(2) != 0 : true;
    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));

    gqaCuda(query, key, value, output, numHeads, numKvHeads, scale, isCausal);
    return sd::Status::OK;
}

PLATFORM_CHECK(grouped_query_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GQA OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, key, value, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(key->dataType()) &&
               llamacppUtils::isSupportedType(value->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// FLASH_ATTENTION - Flash Attention on CUDA
static void flashAttentionCuda(NDArray* query, NDArray* key, NDArray* value,
                               NDArray* output, float scale, bool isCausal) {
    llamacppUtils::GgmlCudaContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, key, ctx.getBackend(), "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, value, ctx.getBackend(), "value");

    // Use GGML's flash attention if available
    struct ggml_tensor* ggml_output = ggml_flash_attn_ext(ctx, ggml_q, ggml_k, ggml_v,
                                                          nullptr, scale, 0.0f, 0.0f);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(flash_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) return sd::Status::OK;

    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / sd::math::sd_sqrt<float, float>(static_cast<float>(query->sizeAt(-1)));
    bool isCausal = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : true;

    flashAttentionCuda(query, key, value, output, scale, isCausal);
    return sd::Status::OK;
}

PLATFORM_CHECK(flash_attention, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA FLASH_ATTENTION OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([query, key, value, output] {
        return llamacppUtils::isSupportedType(query->dataType()) &&
               llamacppUtils::isSupportedType(key->dataType()) &&
               llamacppUtils::isSupportedType(value->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
