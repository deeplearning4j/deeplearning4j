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
// Convolution and pooling operations using GGML CUDA kernels:
// conv2d, depthwise_conv2d, maxpool2d, avgpool2d, im2col
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

//////////////////////////////////////////////////////////////////////////
// CONV_2D - 2D Convolution on CUDA
static void conv2dCuda(NDArray* input, NDArray* kernel, NDArray* bias, NDArray* output,
                       int strideH, int strideW, int padH, int padW, int dilationH, int dilationW) {
    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_kernel = llamacppUtils::createGgmlTensorCuda(ctx, kernel, ctx.getBackend(), "kernel");

    struct ggml_tensor* ggml_output = ggml_conv_2d(ctx, ggml_kernel, ggml_input,
                                                    strideW, strideH, padW, padH, dilationW, dilationH);

    if (bias != nullptr && !bias->isEmpty()) {
        struct ggml_tensor* ggml_bias = llamacppUtils::createGgmlTensorCuda(ctx, bias, ctx.getBackend(), "bias");
        ggml_output = ggml_add(ctx, ggml_output, ggml_bias);
    }

    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(conv2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto kernel = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;
    int pH = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int pW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;
    int dH = block.getIArguments()->size() > 6 ? INT_ARG(6) : 1;
    int dW = block.getIArguments()->size() > 7 ? INT_ARG(7) : 1;

    conv2dCuda(input, kernel, bias, output, sH, sW, pH, pW, dH, dW);
    return sd::Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto kernel = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA CONV2D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, kernel, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(kernel->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectEq(makeInfoVariable(kernel->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DEPTHWISE_CONV2D - Depthwise 2D Convolution on CUDA
static void depthwiseConv2dCuda(NDArray* input, NDArray* kernel, NDArray* bias, NDArray* output,
                                 int strideH, int strideW, int padH, int padW, int dilationH, int dilationW) {
    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_kernel = llamacppUtils::createGgmlTensorCuda(ctx, kernel, ctx.getBackend(), "kernel");

    struct ggml_tensor* ggml_output = ggml_conv_2d_dw(ctx, ggml_kernel, ggml_input,
                                                       strideW, strideH, padW, padH, dilationW, dilationH);

    if (bias != nullptr && !bias->isEmpty()) {
        struct ggml_tensor* ggml_bias = llamacppUtils::createGgmlTensorCuda(ctx, bias, ctx.getBackend(), "bias");
        ggml_output = ggml_add(ctx, ggml_output, ggml_bias);
    }

    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(depthwise_conv2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto kernel = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;
    int pH = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int pW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;
    int dH = block.getIArguments()->size() > 6 ? INT_ARG(6) : 1;
    int dW = block.getIArguments()->size() > 7 ? INT_ARG(7) : 1;

    depthwiseConv2dCuda(input, kernel, bias, output, sH, sW, pH, pW, dH, dW);
    return sd::Status::OK;
}

PLATFORM_CHECK(depthwise_conv2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto kernel = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA DEPTHWISE_CONV2D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, kernel, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(kernel->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// POOL_2D helper for CUDA
static void pool2dCuda(NDArray* input, NDArray* output, int kH, int kW,
                       int sH, int sW, int pH, int pW, int poolType) {
    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    enum ggml_op_pool pool_op = (poolType == 0) ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;

    struct ggml_tensor* ggml_output = ggml_pool_2d(ctx, ggml_input, pool_op, kW, kH, sW, sH, 0.0f);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(maxpool2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;
    int pH = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int pW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;

    pool2dCuda(input, output, kH, kW, sH, sW, pH, pW, 0);  // 0 = max pool
    return sd::Status::OK;
}

PLATFORM_CHECK(maxpool2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MAXPOOL2D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

PLATFORM_IMPL(avgpool2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 2;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;
    int pH = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int pW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;

    pool2dCuda(input, output, kH, kW, sH, sW, pH, pW, 1);  // 1 = avg pool
    return sd::Status::OK;
}

PLATFORM_CHECK(avgpool2d, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA AVGPOOL2D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// IM2COL - Image to Column on CUDA
static void im2colCuda(NDArray* input, NDArray* output, int kH, int kW,
                       int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode) {
    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_im2col(ctx, ggml_input, nullptr,
                                                   kW, kH, sW, sH, pW, pH, dW, dH, isSameMode);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(im2col, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = block.getIArguments()->size() > 0 ? INT_ARG(0) : 3;
    int kW = block.getIArguments()->size() > 1 ? INT_ARG(1) : 3;
    int sH = block.getIArguments()->size() > 2 ? INT_ARG(2) : 1;
    int sW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 1;
    int pH = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int pW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;
    int dH = block.getIArguments()->size() > 6 ? INT_ARG(6) : 1;
    int dW = block.getIArguments()->size() > 7 ? INT_ARG(7) : 1;
    bool isSameMode = block.getIArguments()->size() > 8 ? INT_ARG(8) != 0 : false;

    im2colCuda(input, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode);
    return sd::Status::OK;
}

PLATFORM_CHECK(im2col, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA IM2COL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
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

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
