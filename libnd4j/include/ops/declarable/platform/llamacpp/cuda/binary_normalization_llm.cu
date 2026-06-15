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
// Binary operations, normalization, and LLM-specific operations using GGML CUDA kernels
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

#include <ggml-cuda.h>
#include <ggml-backend.h>

namespace sd {
namespace ops {
namespace platforms {

// Helper macro for binary operations on CUDA
#define DEFINE_CUDA_BINARY_OP(OP_NAME, GGML_FUNC) \
static void OP_NAME##Cuda(NDArray* x, NDArray* y, NDArray* z) { \
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024); \
    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensorCuda(ctx, x, ctx.getBackend(), "x"); \
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensorCuda(ctx, y, ctx.getBackend(), "y"); \
    struct ggml_tensor* ggml_z = GGML_FUNC(ctx, ggml_x, ggml_y); \
    ggml_set_name(ggml_z, "z"); \
    struct ggml_cgraph* graph = ggml_new_graph(ctx); \
    ggml_build_forward_expand(graph, ggml_z); \
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend()); \
    llamacppUtils::copyGgmlCudaToNDArray(ggml_z, z, ctx.getBackend()); \
} \
\
PLATFORM_IMPL(OP_NAME, ENGINE_CUDA) { \
    auto x = INPUT_VARIABLE(0); \
    auto y = INPUT_VARIABLE(1); \
    auto z = OUTPUT_VARIABLE(0); \
    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK; \
    OP_NAME##Cuda(x, y, z); \
    return sd::Status::OK; \
} \
\
PLATFORM_CHECK(OP_NAME, ENGINE_CUDA) { \
    auto x = INPUT_VARIABLE(0); \
    auto y = INPUT_VARIABLE(1); \
    auto z = OUTPUT_VARIABLE(0); \
    Requirements req("LLAMACPP CUDA " #OP_NAME " OP"); \
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG); \
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG); \
    req.expectTrue(makeInfoVariable([x, y, z] { \
        return llamacppUtils::isSupportedType(x->dataType()) && \
               llamacppUtils::isSupportedType(y->dataType()) && \
               llamacppUtils::isSupportedType(z->dataType()); \
    }, TYPECHECK_MSG), NO_MSG); \
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4); \
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4); \
    req.logTheSuccess(); \
    return req; \
}

//////////////////////////////////////////////////////////////////////////
// Binary Operations

DEFINE_CUDA_BINARY_OP(add, ggml_add)
DEFINE_CUDA_BINARY_OP(subtract, ggml_sub)
DEFINE_CUDA_BINARY_OP(multiply, ggml_mul)
DEFINE_CUDA_BINARY_OP(divide, ggml_div)

#undef DEFINE_CUDA_BINARY_OP

//////////////////////////////////////////////////////////////////////////
// TENSORMMUL - Outer product on CUDA
static void tensormmulCuda(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensorCuda(ctx, x, ctx.getBackend(), "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensorCuda(ctx, y, ctx.getBackend(), "y");

    struct ggml_tensor* ggml_z = ggml_out_prod(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_z, z, ctx.getBackend());
}

PLATFORM_IMPL(tensormmul, ENGINE_CUDA) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    tensormmulCuda(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(tensormmul, ENGINE_CUDA) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA TENSORMMUL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Normalization Operations

// GROUP_NORM on CUDA
static void groupNormCuda(NDArray* input, NDArray* output, int numGroups, float eps) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_group_norm(ctx, ggml_input, numGroups, eps);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(group_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int numGroups = block.getIArguments()->size() > 0 ? INT_ARG(0) : 32;
    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    groupNormCuda(input, output, numGroups, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(group_norm, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GROUP_NORM OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 2);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

// L2_NORMALIZE on CUDA
static void l2NormalizeCuda(NDArray* input, NDArray* output, float eps) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_l2_norm(ctx, ggml_input, eps);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(l2_normalize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-12f;

    l2NormalizeCuda(input, output, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(l2_normalize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA L2_NORMALIZE OP");
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

// STANDARDIZE on CUDA
static void standardizeCuda(NDArray* input, NDArray* output, float eps) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_norm(ctx, ggml_input, eps);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(standardize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    standardizeCuda(input, output, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(standardize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA STANDARDIZE OP");
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
// LLM-Specific Operations

// TIMESTEP_EMBEDDING on CUDA
static void timestepEmbeddingCuda(NDArray* timesteps, NDArray* output, int dim, int maxPeriod) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_timesteps = llamacppUtils::createGgmlTensorCuda(ctx, timesteps, ctx.getBackend(), "timesteps");

    struct ggml_tensor* ggml_output = ggml_timestep_embedding(ctx, ggml_timesteps, dim, maxPeriod);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(timestep_embedding, ENGINE_CUDA) {
    auto timesteps = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (timesteps->isEmpty()) return sd::Status::OK;

    int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 128;
    int maxPeriod = block.getIArguments()->size() > 1 ? INT_ARG(1) : 10000;

    timestepEmbeddingCuda(timesteps, output, dim, maxPeriod);
    return sd::Status::OK;
}

PLATFORM_CHECK(timestep_embedding, ENGINE_CUDA) {
    auto timesteps = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA TIMESTEP_EMBEDDING OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([timesteps, output] {
        return llamacppUtils::isSupportedType(timesteps->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(timesteps->rankOf(), RANK_MSG_INPUT0), 1);
    req.logTheSuccess();
    return req;
}

// REGLU on CUDA
static void regluCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_reglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(reglu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    regluCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(reglu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA REGLU OP");
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

// SWIGLU on CUDA
static void swigluCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_swiglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(swiglu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    swigluCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(swiglu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SWIGLU OP");
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

// GATED_LINEAR_ATTN on CUDA
static void gatedLinearAttnCuda(NDArray* query, NDArray* key, NDArray* value,
                                 NDArray* gate, NDArray* output, float scale) {
    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, query, ctx.getBackend(), "query");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, key, ctx.getBackend(), "key");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, value, ctx.getBackend(), "value");
    struct ggml_tensor* ggml_g = gate ? llamacppUtils::createGgmlTensorCuda(ctx, gate, ctx.getBackend(), "gate") : nullptr;

    struct ggml_tensor* ggml_output = ggml_gated_linear_attn(ctx, ggml_q, ggml_k, ggml_v, ggml_g, scale);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(gated_linear_attn, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    NDArray* gate = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (query->isEmpty() || key->isEmpty() || value->isEmpty()) {
        return sd::Status::OK;
    }

    float scale = block.getTArguments()->size() > 0 ?
        T_ARG(0) : 1.0f / std::sqrt(static_cast<float>(query->sizeAt(-1)));

    gatedLinearAttnCuda(query, key, value, gate, output, scale);
    return sd::Status::OK;
}

PLATFORM_CHECK(gated_linear_attn, ENGINE_CUDA) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GATED_LINEAR_ATTN OP");
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

// FILL on CUDA
static void fillCuda(NDArray* output, float value) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

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
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(fill, ENGINE_CUDA) {
    auto output = OUTPUT_VARIABLE(0);

    float value = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;

    fillCuda(output, value);
    return sd::Status::OK;
}

PLATFORM_CHECK(fill, ENGINE_CUDA) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA FILL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT0), 4);
    req.logTheSuccess();
    return req;
}

// EXPM1 on CUDA
static void expm1Cuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_expm1(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(expm1, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    expm1Cuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(expm1, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA EXPM1 OP");
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

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
