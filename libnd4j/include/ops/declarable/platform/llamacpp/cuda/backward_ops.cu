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
// CUDA implementations of backward/gradient operations using GGML kernels
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
// SILU_BP - SiLU backward pass (CUDA)
PLATFORM_IMPL(silu_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_silu_back(ctx, ggml_input, ggml_grad);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(silu_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SILU_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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
// RMS_NORM_BP - RMS normalization backward (CUDA)
PLATFORM_IMPL(rms_norm_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_rms_norm_back(ctx, ggml_input, ggml_grad, eps);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(rms_norm_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA RMS_NORM_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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
// ROPE_BP - RoPE backward (CUDA)
PLATFORM_IMPL(rope_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int mode = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int nDims = block.getIArguments()->size() > 1 ? INT_ARG(1) : input->sizeAt(-1);
    int nCtx = block.getIArguments()->size() > 2 ? INT_ARG(2) : 2048;
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_rope_back(ctx, ggml_grad, nullptr, nDims, mode,
                                                      nCtx, 0, freqBase, freqScale, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(rope_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ROPE_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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
// SOFTMAX_BP - Softmax backward (CUDA)
PLATFORM_IMPL(softmax_bp, ENGINE_CUDA) {
    auto output = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (output->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_output = llamacppUtils::createGgmlTensorCuda(ctx, output, ctx.getBackend(), "output");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_soft_max_back(ctx, ggml_output, ggml_grad);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(softmax_bp, ENGINE_CUDA) {
    auto output = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SOFTMAX_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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
// MAXPOOL2D_BP - Max pooling 2D backward (CUDA)
PLATFORM_IMPL(maxpool2d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_pool_2d_back(ctx, ggml_grad, ggml_input, GGML_OP_POOL_MAX, kW, kH, sW, sH, 0.0f);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(maxpool2d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MAXPOOL2D_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// AVGPOOL2D_BP - Average pooling 2D backward (CUDA)
PLATFORM_IMPL(avgpool2d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_result = ggml_pool_2d_back(ctx, ggml_grad, ggml_input, GGML_OP_POOL_AVG, kW, kH, sW, sH, 0.0f);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(avgpool2d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA AVGPOOL2D_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
