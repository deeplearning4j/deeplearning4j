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
// Activation functions using GGML kernels:
// relu, elu, sigmoid, hardswish, hardsigmoid, leaky_relu, softplus, step
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
// RELU - Rectified Linear Unit
static void reluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_relu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    reluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP RELU OP");
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
// ELU - Exponential Linear Unit
static void eluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_elu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    eluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ELU OP");
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
// SIGMOID
static void sigmoidLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_sigmoid(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    sigmoidLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SIGMOID OP");
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
// HARDSWISH - Hard Swish activation
// hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
static void hardswishLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_hardswish(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(hardswish, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    hardswishLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(hardswish, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP HARDSWISH OP");
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
// HARDSIGMOID - Hard Sigmoid activation
// hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
static void hardsigmoidLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_hardsigmoid(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(hardsigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    hardsigmoidLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(hardsigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP HARDSIGMOID OP");
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
// LEAKY_RELU - Leaky Rectified Linear Unit
// leaky_relu(x, alpha) = x if x >= 0, alpha * x otherwise
static void leakyReluLlamaCpp(NDArray* input, NDArray* output, float negativeSlope) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_leaky_relu(ctx, ggml_input, negativeSlope, false);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(leakyrelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float negativeSlope = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.01f;
    leakyReluLlamaCpp(input, output, negativeSlope);
    return sd::Status::OK;
}

PLATFORM_CHECK(leakyrelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP LEAKYRELU OP");
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
// SOFTPLUS - Softplus activation
// softplus(x) = log(1 + exp(x))
static void softplusLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_softplus(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(softplus, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    softplusLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(softplus, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SOFTPLUS OP");
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
// STEP - Step function (Heaviside)
// step(x) = 1 if x > 0, 0 otherwise
static void stepLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_step(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(step, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    stepLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(step, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP STEP OP");
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
// GELU_ERF - Exact GELU using error function
// More accurate than GELU_QUICK but slower
static void geluErfLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_gelu_erf(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(gelu_erf, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    geluErfLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(gelu_erf, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GELU_ERF OP");
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
