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
// Embedding and positional encoding operations using GGML kernels
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
// EMBEDDING_LOOKUP - Look up embeddings from weight matrix
static void embeddingLookupLlamaCpp(NDArray* weights, NDArray* indices, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensor(ctx, weights, "weights");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, indices, "indices");

    struct ggml_tensor* ggml_output = ggml_get_rows(ctx, ggml_weights, ggml_indices);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(embedding_lookup, ENGINE_CPU) {
    auto weights = INPUT_VARIABLE(0);  // [vocab_size, embed_dim]
    auto indices = INPUT_VARIABLE(1);  // [batch, seq_len]
    auto output = OUTPUT_VARIABLE(0);  // [batch, seq_len, embed_dim]

    if (weights->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    embeddingLookupLlamaCpp(weights, indices, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(embedding_lookup, ENGINE_CPU) {
    auto weights = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP EMBEDDING_LOOKUP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([weights, output] {
        return llamacppUtils::isSupportedType(weights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// EMBEDDING_LOOKUP_BP - Embedding lookup backward (gradient accumulation)
static void embeddingLookupBpLlamaCpp(NDArray* gradOutput, NDArray* indices, NDArray* gradWeights) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, indices, "indices");

    // Accumulate gradients back into weight gradients
    struct ggml_tensor* ggml_output = ggml_acc(ctx,
        ggml_new_tensor(ctx, ggml_grad->type, gradWeights->rankOf(), gradWeights->shapeOf()),
        ggml_grad, 0, 0, 0, 0);
    ggml_set_name(ggml_output, "grad_weights");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, gradWeights);
}

PLATFORM_IMPL(embedding_lookup_bp, ENGINE_CPU) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradWeights = OUTPUT_VARIABLE(0);

    if (gradOutput->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    embeddingLookupBpLlamaCpp(gradOutput, indices, gradWeights);
    return sd::Status::OK;
}

PLATFORM_CHECK(embedding_lookup_bp, ENGINE_CPU) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradWeights = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP EMBEDDING_LOOKUP_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([gradOutput, gradWeights] {
        return llamacppUtils::isSupportedType(gradOutput->dataType()) &&
               llamacppUtils::isSupportedType(gradWeights->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SINUSOIDAL_POSITION_ENCODING - Generate sinusoidal positional embeddings
static void sinusoidalPositionEncodingLlamaCpp(NDArray* positions, NDArray* output, int embedDim) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_positions = llamacppUtils::createGgmlTensor(ctx, positions, "positions");

    // Apply RoPE-style positional encoding (sinusoidal)
    struct ggml_tensor* ggml_output = ggml_rope(ctx, ggml_positions, nullptr, embedDim, 0, 2048, 10000.0f);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(sinusoidal_position_encoding, ENGINE_CPU) {
    auto positions = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (positions->isEmpty()) return sd::Status::OK;

    int embedDim = block.getIArguments()->size() > 0 ? INT_ARG(0) : output->sizeAt(-1);

    sinusoidalPositionEncodingLlamaCpp(positions, output, embedDim);
    return sd::Status::OK;
}

PLATFORM_CHECK(sinusoidal_position_encoding, ENGINE_CPU) {
    auto positions = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SINUSOIDAL_POSITION_ENCODING OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([positions, output] {
        return llamacppUtils::isSupportedType(positions->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(positions->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ALIBI_POSITION_BIAS - Attention with Linear Biases (ALiBi)
static void alibiPositionBiasLlamaCpp(NDArray* attentionScores, NDArray* output,
                                       int numHeads, float slope) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_scores = llamacppUtils::createGgmlTensor(ctx, attentionScores, "scores");

    // Apply ALiBi bias
    struct ggml_tensor* ggml_output = ggml_alibi(ctx, ggml_scores, 0, numHeads, slope);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(alibi_position_bias, ENGINE_CPU) {
    auto attentionScores = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (attentionScores->isEmpty()) return sd::Status::OK;

    int numHeads = block.getIArguments()->size() > 0 ? INT_ARG(0) : attentionScores->sizeAt(1);
    float slope = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;

    alibiPositionBiasLlamaCpp(attentionScores, output, numHeads, slope);
    return sd::Status::OK;
}

PLATFORM_CHECK(alibi_position_bias, ENGINE_CPU) {
    auto attentionScores = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ALIBI_POSITION_BIAS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([attentionScores, output] {
        return llamacppUtils::isSupportedType(attentionScores->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(attentionScores->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GET_ROWS - Get rows from matrix (same as embedding lookup but more general)
static void getRowsLlamaCpp(NDArray* input, NDArray* indices, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, indices, "indices");

    struct ggml_tensor* ggml_output = ggml_get_rows(ctx, ggml_input, ggml_indices);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(get_rows, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    getRowsLlamaCpp(input, indices, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(get_rows, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GET_ROWS OP");
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
// GET_ROWS_BACK - Backward pass for get_rows
static void getRowsBackLlamaCpp(NDArray* gradOutput, NDArray* indices, NDArray* gradInput) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_grad = llamacppUtils::createGgmlTensor(ctx, gradOutput, "grad_output");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, indices, "indices");
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, gradInput, "grad_input");

    struct ggml_tensor* ggml_output = ggml_get_rows_back(ctx, ggml_grad, ggml_indices, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, gradInput);
}

PLATFORM_IMPL(get_rows_bp, ENGINE_CPU) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    if (gradOutput->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    getRowsBackLlamaCpp(gradOutput, indices, gradInput);
    return sd::Status::OK;
}

PLATFORM_CHECK(get_rows_bp, ENGINE_CPU) {
    auto gradOutput = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GET_ROWS_BP OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
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

#endif  // HAVE_LLAMACPP
