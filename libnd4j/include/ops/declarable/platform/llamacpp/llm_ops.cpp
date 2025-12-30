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
// LLM-specific operations using GGML kernels:
// timestep_embedding, gated_linear_attn, reglu, swiglu, fill
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
// TIMESTEP_EMBEDDING - Sinusoidal timestep embedding for diffusion models
static void timestepEmbeddingLlamaCpp(NDArray* timesteps, NDArray* output, int dim, int maxPeriod) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_timesteps = llamacppUtils::createGgmlTensor(ctx, timesteps, "timesteps");

    struct ggml_tensor* ggml_output = ggml_timestep_embedding(ctx, ggml_timesteps, dim, maxPeriod);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(timestep_embedding, ENGINE_CPU) {
    auto timesteps = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (timesteps->isEmpty()) return sd::Status::OK;

    int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 128;
    int maxPeriod = block.getIArguments()->size() > 1 ? INT_ARG(1) : 10000;

    timestepEmbeddingLlamaCpp(timesteps, output, dim, maxPeriod);
    return sd::Status::OK;
}

PLATFORM_CHECK(timestep_embedding, ENGINE_CPU) {
    auto timesteps = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP TIMESTEP_EMBEDDING OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([timesteps, output] {
        return llamacppUtils::isSupportedType(timesteps->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    // Timesteps should be 1D
    req.expectEq(makeInfoVariable(timesteps->rankOf(), RANK_MSG_INPUT0), 1);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// REGLU - Gated Linear Unit with ReLU
// ReGLU(x, W, V, b, c) = ReLU(xW + b) * (xV + c)
static void regluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_reglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(reglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    regluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(reglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP REGLU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SWIGLU - Gated Linear Unit with SiLU/Swish
// SwiGLU(x, W, V, b, c) = SiLU(xW + b) * (xV + c)
// Used in LLaMA, PaLM, and other modern LLMs
static void swigluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_swiglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(swiglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    swigluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(swiglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SWIGLU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GATED_LINEAR_ATTN - Gated Linear Attention (for linear attention variants)
static void gatedLinearAttnLlamaCpp(NDArray* query, NDArray* key, NDArray* value,
                                     NDArray* gate, NDArray* output, float scale) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, query, "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, key, "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, value, "value");
    struct ggml_tensor* ggml_g = gate ? llamacppUtils::createGgmlTensor(ctx, gate, "gate") : nullptr;

    struct ggml_tensor* ggml_output = ggml_gated_linear_attn(ctx, ggml_q, ggml_k, ggml_v, ggml_g, scale);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(gated_linear_attn, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    NDArray* gate = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) {
        return sd::Status::OK;
    }

    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) :
        1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    gatedLinearAttnLlamaCpp(query, key, value, gate, output, scale);
    return sd::Status::OK;
}

PLATFORM_CHECK(gated_linear_attn, ENGINE_CPU) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GATED_LINEAR_ATTN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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
// FILL - Fill tensor with a constant value
static void fillLlamaCpp(NDArray* output, float value) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    // Create output shape tensor
    const auto shape = output->shapeOf();
    const auto rank = output->rankOf();

    struct ggml_tensor* ggml_output;
    if (rank == 1) {
        ggml_output = ggml_new_tensor_1d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[0]);
    } else if (rank == 2) {
        ggml_output = ggml_new_tensor_2d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[1], shape[0]);
    } else if (rank == 3) {
        ggml_output = ggml_new_tensor_3d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[2], shape[1], shape[0]);
    } else {
        ggml_output = ggml_new_tensor_4d(ctx, llamacppUtils::toGgmlType(output->dataType()),
            rank >= 4 ? shape[3] : 1, rank >= 3 ? shape[2] : 1, shape[1], shape[0]);
    }

    ggml_output = ggml_fill(ctx, ggml_output, value);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(fill, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    float value = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;

    fillLlamaCpp(output, value);
    return sd::Status::OK;
}

PLATFORM_CHECK(fill, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP FILL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// EXPM1 - exp(x) - 1 (more accurate for small x)
static void expm1LlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_expm1(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(expm1, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    expm1LlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(expm1, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP EXPM1 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
