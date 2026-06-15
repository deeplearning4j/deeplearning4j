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
// Activation functions using GGML CUDA kernels:
// relu, elu, sigmoid, hardswish, hardsigmoid, leakyrelu, softplus, step, gelu_erf
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

// Helper macro for simple unary activation operations on CUDA
#define DEFINE_CUDA_ACTIVATION_OP(OP_NAME, GGML_FUNC) \
static void OP_NAME##Cuda(NDArray* input, NDArray* output) { \
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024); \
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input"); \
    struct ggml_tensor* ggml_output = GGML_FUNC(ctx, ggml_input); \
    ggml_set_name(ggml_output, "output"); \
    struct ggml_cgraph* graph = ggml_new_graph(ctx); \
    ggml_build_forward_expand(graph, ggml_output); \
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend()); \
    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend()); \
} \
\
PLATFORM_IMPL(OP_NAME, ENGINE_CUDA) { \
    auto input = INPUT_VARIABLE(0); \
    auto output = OUTPUT_VARIABLE(0); \
    if (input->isEmpty()) return sd::Status::OK; \
    OP_NAME##Cuda(input, output); \
    return sd::Status::OK; \
} \
\
PLATFORM_CHECK(OP_NAME, ENGINE_CUDA) { \
    auto input = INPUT_VARIABLE(0); \
    auto output = OUTPUT_VARIABLE(0); \
    Requirements req("LLAMACPP CUDA " #OP_NAME " OP"); \
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG); \
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG); \
    req.expectTrue(makeInfoVariable([input, output] { \
        return llamacppUtils::isSupportedType(input->dataType()) && \
               llamacppUtils::isSupportedType(output->dataType()); \
    }, TYPECHECK_MSG), NO_MSG); \
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4); \
    req.logTheSuccess(); \
    return req; \
}

//////////////////////////////////////////////////////////////////////////
// Define simple activation operations

DEFINE_CUDA_ACTIVATION_OP(relu, ggml_relu)
DEFINE_CUDA_ACTIVATION_OP(sigmoid, ggml_sigmoid)
DEFINE_CUDA_ACTIVATION_OP(hardswish, ggml_hardswish)
DEFINE_CUDA_ACTIVATION_OP(hardsigmoid, ggml_hardsigmoid)
DEFINE_CUDA_ACTIVATION_OP(softplus, ggml_softplus)
DEFINE_CUDA_ACTIVATION_OP(step, ggml_step)
DEFINE_CUDA_ACTIVATION_OP(gelu_erf, ggml_gelu)

#undef DEFINE_CUDA_ACTIVATION_OP

//////////////////////////////////////////////////////////////////////////
// ELU - Exponential Linear Unit (with alpha parameter)
static void eluCuda(NDArray* input, NDArray* output, float alpha) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_elu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(elu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;
    eluCuda(input, output, alpha);
    return sd::Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ELU OP");
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
// LEAKY_RELU - Leaky ReLU (with negative slope parameter)
static void leakyReluCuda(NDArray* input, NDArray* output, float negativeSlope) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_leaky_relu(ctx, ggml_input, negativeSlope, false);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(leakyrelu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float negativeSlope = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.01f;
    leakyReluCuda(input, output, negativeSlope);
    return sd::Status::OK;
}

PLATFORM_CHECK(leakyrelu, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA LEAKYRELU OP");
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
