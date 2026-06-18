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
// Mixture of Experts (MoE) operations using GGML kernels
// Supports sparse and dense MoE architectures (Mixtral, etc.)
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
// MOE_GATE - Compute routing weights for MoE
// Takes input and gating weights, produces expert selection probabilities
static void moeGateLlamaCpp(NDArray* input, NDArray* gateWeights, NDArray* output, int topK) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_gate = llamacppUtils::createGgmlTensor(ctx, gateWeights, "gate");

    // Compute gate logits: input @ gate.T
    struct ggml_tensor* ggml_logits = ggml_mul_mat(ctx, ggml_gate, ggml_input);

    // Apply softmax to get routing probabilities
    struct ggml_tensor* ggml_probs = ggml_soft_max(ctx, ggml_logits);

    // Get top-k experts
    struct ggml_tensor* ggml_output = ggml_top_k(ctx, ggml_probs, topK);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(moe_gate, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gateWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || gateWeights->isEmpty()) return sd::Status::OK;

    int topK = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;

    moeGateLlamaCpp(input, gateWeights, output, topK);
    return sd::Status::OK;
}

PLATFORM_CHECK(moe_gate, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gateWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP MOE_GATE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, gateWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(gateWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MOE_EXPERT_FFN - Execute expert FFN with gating
// Combines expert outputs weighted by routing probabilities
static void moeExpertFfnLlamaCpp(NDArray* input, NDArray* expertWeights, NDArray* routingWeights,
                                  NDArray* expertIndices, NDArray* output, int numExperts) {
    llamacppUtils::GgmlContextGuard ctx(256 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_experts = llamacppUtils::createGgmlTensor(ctx, expertWeights, "experts");
    struct ggml_tensor* ggml_routing = llamacppUtils::createGgmlTensor(ctx, routingWeights, "routing");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, expertIndices, "indices");

    // For each selected expert, compute FFN and weight by routing probability
    // This is a simplified implementation - full MoE would use sparse dispatch
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_experts, ggml_input);

    // Apply routing weights
    ggml_output = ggml_mul(ctx, ggml_output, ggml_routing);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(moe_expert_ffn, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto expertWeights = INPUT_VARIABLE(1);
    auto routingWeights = INPUT_VARIABLE(2);
    auto expertIndices = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int numExperts = block.getIArguments()->size() > 0 ? INT_ARG(0) : 8;

    moeExpertFfnLlamaCpp(input, expertWeights, routingWeights, expertIndices, output, numExperts);
    return sd::Status::OK;
}

PLATFORM_CHECK(moe_expert_ffn, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto expertWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP MOE_EXPERT_FFN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, expertWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(expertWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SPARSE_MUL_MAT - Sparse matrix multiplication
// Used for efficient MoE computation with sparse expert selection
static void sparseMulMatLlamaCpp(NDArray* input, NDArray* sparseWeights, NDArray* indices, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensor(ctx, sparseWeights, "weights");
    struct ggml_tensor* ggml_indices = llamacppUtils::createGgmlTensor(ctx, indices, "indices");

    // Get selected rows and multiply
    struct ggml_tensor* ggml_selected = ggml_get_rows(ctx, ggml_weights, ggml_indices);
    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_selected, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(sparse_mul_mat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto sparseWeights = INPUT_VARIABLE(1);
    auto indices = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || sparseWeights->isEmpty()) return sd::Status::OK;

    sparseMulMatLlamaCpp(input, sparseWeights, indices, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(sparse_mul_mat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto sparseWeights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SPARSE_MUL_MAT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, sparseWeights, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(sparseWeights->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 3);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// LOAD_BALANCE_LOSS - Compute load balancing loss for MoE training
// Encourages uniform expert utilization
PLATFORM_IMPL(load_balance_loss, ENGINE_CPU) {
    auto routingProbs = INPUT_VARIABLE(0);  // [batch, num_experts]
    auto expertMask = INPUT_VARIABLE(1);    // [batch, num_experts] - one-hot selected experts
    auto output = OUTPUT_VARIABLE(0);       // scalar loss

    if (routingProbs->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_probs = llamacppUtils::createGgmlTensor(ctx, routingProbs, "probs");
    struct ggml_tensor* ggml_mask = llamacppUtils::createGgmlTensor(ctx, expertMask, "mask");

    // Compute mean probability per expert
    struct ggml_tensor* ggml_mean_probs = ggml_mean(ctx, ggml_probs);

    // Compute fraction of tokens assigned to each expert
    struct ggml_tensor* ggml_mean_mask = ggml_mean(ctx, ggml_mask);

    // Load balance loss = num_experts * sum(mean_probs * mean_mask)
    struct ggml_tensor* ggml_product = ggml_mul(ctx, ggml_mean_probs, ggml_mean_mask);
    struct ggml_tensor* ggml_output = ggml_sum(ctx, ggml_product);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(load_balance_loss, ENGINE_CPU) {
    auto routingProbs = INPUT_VARIABLE(0);
    auto expertMask = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP LOAD_BALANCE_LOSS OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([routingProbs, expertMask, output] {
        return llamacppUtils::isSupportedType(routingProbs->dataType()) &&
               llamacppUtils::isSupportedType(expertMask->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(routingProbs->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
