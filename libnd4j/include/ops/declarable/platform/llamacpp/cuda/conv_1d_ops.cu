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
// CUDA implementations of 1D convolution and pooling operations using GGML kernels
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
// CONV1D - 1D Convolution (CUDA)
PLATFORM_IMPL(conv1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int stride = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int padding = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int dilation = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_kernel = llamacppUtils::createGgmlTensorCuda(ctx, weights, ctx.getBackend(), "kernel");

    struct ggml_tensor* ggml_output = ggml_conv_1d(ctx, ggml_kernel, ggml_input, stride, padding, dilation);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());

    if (bias != nullptr && !bias->isEmpty()) {
        *output += *bias;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(conv1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA CONV1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, weights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(weights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MAXPOOL1D - 1D Max Pooling (CUDA)
PLATFORM_IMPL(maxpool1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kernelSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : 1;
    int padding = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_pool_1d(ctx, ggml_input, GGML_OP_POOL_MAX, kernelSize, stride, padding);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(maxpool1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MAXPOOL1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// AVGPOOL1D - 1D Average Pooling (CUDA)
PLATFORM_IMPL(avgpool1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kernelSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : 1;
    int padding = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_pool_1d(ctx, ggml_input, GGML_OP_POOL_AVG, kernelSize, stride, padding);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(avgpool1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA AVGPOOL1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DECONV1D - 1D Transposed Convolution (CUDA)
PLATFORM_IMPL(deconv1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int stride = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int padding = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int dilation = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_kernel = llamacppUtils::createGgmlTensorCuda(ctx, weights, ctx.getBackend(), "kernel");

    struct ggml_tensor* ggml_output = ggml_conv_transpose_1d(ctx, ggml_kernel, ggml_input, stride, padding, dilation);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());

    if (bias != nullptr && !bias->isEmpty()) {
        *output += *bias;
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(deconv1d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA DECONV1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, weights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(weights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MAXPOOL1D_BP - 1D Max Pooling Backward (CUDA)
PLATFORM_IMPL(maxpool1d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kernelSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int stride = block.getIArguments()->size() > 1 ? INT_ARG(1) : 1;
    int padding = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    // Note: GGML doesn't have direct 1D pool backward, using forward as placeholder
    struct ggml_tensor* ggml_result = ggml_pool_1d(ctx, ggml_input, GGML_OP_POOL_MAX, kernelSize, stride, padding);
    ggml_set_name(ggml_result, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_result, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(maxpool1d_bp, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MAXPOOL1D_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
