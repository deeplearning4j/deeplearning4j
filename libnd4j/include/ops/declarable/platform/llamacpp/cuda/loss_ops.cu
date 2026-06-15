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
// CUDA implementations of loss function operations using GGML kernels
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
// SOFTMAX_CROSS_ENTROPY_LOSS - Cross entropy loss (CUDA)
PLATFORM_IMPL(softmax_cross_entropy_loss, ENGINE_CUDA) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (logits->isEmpty() || labels->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_logits = llamacppUtils::createGgmlTensorCuda(ctx, logits, ctx.getBackend(), "logits");
    struct ggml_tensor* ggml_labels = llamacppUtils::createGgmlTensorCuda(ctx, labels, ctx.getBackend(), "labels");

    struct ggml_tensor* ggml_output = ggml_cross_entropy_loss(ctx, ggml_logits, ggml_labels);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(softmax_cross_entropy_loss, ENGINE_CUDA) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SOFTMAX_CROSS_ENTROPY_LOSS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([logits, labels, output] {
        return llamacppUtils::isSupportedType(logits->dataType()) &&
               llamacppUtils::isSupportedType(labels->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(logits->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SOFTMAX_CROSS_ENTROPY_LOSS_BP - Cross entropy loss backward (CUDA)
PLATFORM_IMPL(softmax_cross_entropy_loss_bp, ENGINE_CUDA) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto gradOutput = INPUT_VARIABLE(2);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (logits->isEmpty() || labels->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_logits = llamacppUtils::createGgmlTensorCuda(ctx, logits, ctx.getBackend(), "logits");
    struct ggml_tensor* ggml_labels = llamacppUtils::createGgmlTensorCuda(ctx, labels, ctx.getBackend(), "labels");
    struct ggml_tensor* ggml_grad_out = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");

    struct ggml_tensor* ggml_output = ggml_cross_entropy_loss_back(ctx, ggml_logits, ggml_labels, ggml_grad_out);
    ggml_set_name(ggml_output, "grad_input");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(softmax_cross_entropy_loss_bp, ENGINE_CUDA) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto gradOutput = INPUT_VARIABLE(2);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SOFTMAX_CROSS_ENTROPY_LOSS_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([logits, labels, gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(logits->dataType()) &&
               llamacppUtils::isSupportedType(labels->dataType()) &&
               llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(logits->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
