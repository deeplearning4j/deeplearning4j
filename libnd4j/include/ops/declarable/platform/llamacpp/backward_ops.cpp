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
// Backward/gradient operations using GGML kernels:
// silu_back, rms_norm_back, rope_back, softmax_back
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
// SILU_BACK - SiLU backward pass
static void siluBackLlamaCpp(NDArray* input, NDArray* gradOutput, NDArray* gradInput) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");

    struct ggml_tensor* ggml_result = ggml_silu_back(ctx, ggml_input, ggml_grad);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_result, gradInput);
}

PLATFORM_IMPL(silu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    siluBackLlamaCpp(input, gradOutput, gradInput);
    return sd::Status::OK;
}

PLATFORM_CHECK(silu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SILU_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// RMS_NORM_BACK - RMS normalization backward pass
static void rmsNormBackLlamaCpp(NDArray* input, NDArray* gradOutput, NDArray* gradInput, float eps) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");

    struct ggml_tensor* ggml_result = ggml_rms_norm_back(ctx, ggml_input, ggml_grad, eps);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_result, gradInput);
}

PLATFORM_IMPL(rms_norm_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;
    rmsNormBackLlamaCpp(input, gradOutput, gradInput, eps);
    return sd::Status::OK;
}

PLATFORM_CHECK(rms_norm_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP RMS_NORM_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ROPE_BACK - Rotary Position Embedding backward pass
static void ropeBackLlamaCpp(NDArray* input, NDArray* gradOutput, NDArray* gradInput,
                              int nDims, int mode, int nCtx, float freqBase, float freqScale) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");

    struct ggml_tensor* ggml_result = ggml_rope_back(ctx, ggml_grad, nullptr, nDims, mode,
                                                      nCtx, 0, freqBase, freqScale, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_result, gradInput);
}

PLATFORM_IMPL(rope_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int nDims = block.getIArguments()->size() > 1 ? INT_ARG(1) : input->sizeAt(-1);
    int nCtx = block.getIArguments()->size() > 2 ? INT_ARG(2) : 2048;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    ropeBackLlamaCpp(input, gradOutput, gradInput, nDims, mode, nCtx, freqBase, freqScale);
    return sd::Status::OK;
}

PLATFORM_CHECK(rope_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ROPE_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SOFTMAX_BACK - Softmax backward pass
static void softmaxBackLlamaCpp(NDArray* output, NDArray* gradOutput, NDArray* gradInput) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_output = llamacppUtils::createGgmlTensor(ctx, output, "output");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");

    struct ggml_tensor* ggml_result = ggml_soft_max_back(ctx, ggml_output, ggml_grad);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_result, gradInput);
}

PLATFORM_IMPL(softmax_bp, ENGINE_CPU) {
    auto output = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (output->isEmpty()) return sd::Status::OK;

    softmaxBackLlamaCpp(output, gradOutput, gradInput);
    return sd::Status::OK;
}

PLATFORM_CHECK(softmax_bp, ENGINE_CPU) {
    auto output = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SOFTMAX_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([output, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(output->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// POOL_2D_BACK - 2D Pooling backward pass
static void pool2dBackLlamaCpp(NDArray* input, NDArray* gradOutput, NDArray* gradInput,
                                int kH, int kW, int sH, int sW, int poolType) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");

    enum ggml_op_pool pool_op = (poolType == 0) ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;

    struct ggml_tensor* ggml_result = ggml_pool_2d_back(ctx, ggml_grad, ggml_input, pool_op, kW, kH, sW, sH, 0.0f);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_result, gradInput);
}

PLATFORM_IMPL(maxpool2d_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;

    pool2dBackLlamaCpp(input, gradOutput, gradInput, kH, kW, sH, sW, 0);  // 0 = max pool
    return sd::Status::OK;
}

PLATFORM_CHECK(maxpool2d_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP MAXPOOL2D_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

PLATFORM_IMPL(avgpool2d_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;

    pool2dBackLlamaCpp(input, gradOutput, gradInput, kH, kW, sH, sW, 1);  // 1 = avg pool
    return sd::Status::OK;
}

PLATFORM_CHECK(avgpool2d_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP AVGPOOL2D_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
