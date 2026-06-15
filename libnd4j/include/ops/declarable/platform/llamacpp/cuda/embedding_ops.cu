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
// CUDA implementations of embedding and positional encoding operations
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
// EMBEDDING_LOOKUP - CUDA
PLATFORM_IMPL(embedding_lookup, ENGINE_CUDA) {
    auto weights = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (weights->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensorCuda(ctx, weights, ctx.getBackend(), "weights");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensorCuda(ctx, indices, ctx.getBackend(), "indices");

    struct ggml_tensor* ggml_output = ggml_get_rows(ctx, ggml_weights, ggml_indices);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(embedding_lookup, ENGINE_CUDA) {
    auto weights = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA EMBEDDING_LOOKUP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([weights, output] {
        return llamacppUtils::isSupportedType(weights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// EMBEDDING_LOOKUP_BP - CUDA
PLATFORM_IMPL(embedding_lookup_bp, ENGINE_CUDA) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradWeights = OUTPUT_VARIABLE(0);

    if (gradOutput->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensorCuda(ctx, indices, ctx.getBackend(), "indices");
    struct ggml_tensor* ggml_grad_w = llamacppUtils::createGgmlTensorCuda(ctx, gradWeights, ctx.getBackend(), "grad_weights");

    struct ggml_tensor* ggml_output = ggml_get_rows_back(ctx, ggml_grad, ggml_indices, ggml_grad_w);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, gradWeights, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(embedding_lookup_bp, ENGINE_CUDA) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto gradWeights = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA EMBEDDING_LOOKUP_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([gradOutput, gradWeights] {
        return llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradWeights->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SINUSOIDAL_POSITION_ENCODING - CUDA
PLATFORM_IMPL(sinusoidal_position_encoding, ENGINE_CUDA) {
    auto positions = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (positions->isEmpty()) return sd::Status::OK;

    int embedDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : output->sizeAt(-1);

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_positions = llamacppUtils::createGgmlTensorCuda(ctx, positions, ctx.getBackend(), "positions");
    struct ggml_tensor* ggml_output = ggml_rope(ctx, ggml_positions, nullptr, embedDim, 0, 2048, 10000.0f);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(sinusoidal_position_encoding, ENGINE_CUDA) {
    auto positions = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SINUSOIDAL_POSITION_ENCODING OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([positions, output] {
        return llamacppUtils::isSupportedType(positions->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(positions->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ALIBI_POSITION_BIAS - CUDA
PLATFORM_IMPL(alibi_position_bias, ENGINE_CUDA) {
    auto attentionScores = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (attentionScores->isEmpty()) return sd::Status::OK;

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : attentionScores->sizeAt(1);
    float slope = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_scores = llamacppUtils::createGgmlTensorCuda(ctx, attentionScores, ctx.getBackend(), "scores");
    struct ggml_tensor* ggml_output = ggml_alibi(ctx, ggml_scores, 0, numHeads, slope);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(alibi_position_bias, ENGINE_CUDA) {
    auto attentionScores = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ALIBI_POSITION_BIAS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([attentionScores, output] {
        return llamacppUtils::isSupportedType(attentionScores->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(attentionScores->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GET_ROWS - CUDA
PLATFORM_IMPL(get_rows, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensorCuda(ctx, indices, ctx.getBackend(), "indices");

    struct ggml_tensor* ggml_output = ggml_get_rows(ctx, ggml_input, ggml_indices);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(get_rows, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GET_ROWS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GET_ROWS_BP - CUDA
PLATFORM_IMPL(get_rows_bp, ENGINE_CUDA) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (gradOutput->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensorCuda(ctx, gradOutput, ctx.getBackend(), "grad_output");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensorCuda(ctx, indices, ctx.getBackend(), "indices");
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, gradInput, ctx.getBackend(), "grad_input");

    struct ggml_tensor* ggml_output = ggml_get_rows_back(ctx, ggml_grad, ggml_indices, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, gradInput, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(get_rows_bp, ENGINE_CUDA) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GET_ROWS_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([gradOutput, gradInput] {
        return llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradInput->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
