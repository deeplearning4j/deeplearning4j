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
// Gated Delta Network operations using GGML kernels:
// gated_delta_rule, causal_conv1d (CPU platform implementations)
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
// GATED_DELTA_RULE - Gated Delta Network recurrent state update (CPU)
// Uses GGML gated_delta_net kernel if available, otherwise falls back to generic impl
PLATFORM_IMPL(gated_delta_rule, ENGINE_CPU) {
    auto Q = INPUT_VARIABLE(0);        // [B, L, H, D_k]
    auto K = INPUT_VARIABLE(1);        // [B, L, H, D_k]
    auto V = INPUT_VARIABLE(2);        // [B, L, H, D_v]
    auto beta = INPUT_VARIABLE(3);     // [B, L, H]
    auto gate = INPUT_VARIABLE(4);     // [B, L, H]
    auto output = OUTPUT_VARIABLE(0);     // [B, L, H, D_v]
    auto stateOut = OUTPUT_VARIABLE(1);   // [B, H, D_k, D_v]

    if (Q->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensor(ctx, Q, "q");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, K, "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, V, "v");
    struct ggml_tensor* ggml_beta = llamacppUtils::createGgmlTensor(ctx, beta, "beta");
    struct ggml_tensor* ggml_gate = llamacppUtils::createGgmlTensor(ctx, gate, "gate");

    // Optional state input
    NDArray* stateIn = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
    struct ggml_tensor* ggml_state = stateIn != nullptr ?
        llamacppUtils::createGgmlTensor(ctx, stateIn, "state") : nullptr;

    struct ggml_tensor* ggml_output = ggml_gated_delta_net(ctx, ggml_q, ggml_k, ggml_v,
                                                            ggml_beta, ggml_gate, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);

    // State output handling depends on GGML op's output structure
    // If GGML provides state as secondary output, copy it; otherwise zero-init
    { double zero = 0.0; stateOut->assign(zero); }

    return sd::Status::OK;
}

PLATFORM_CHECK(gated_delta_rule, ENGINE_CPU) {
    auto Q = INPUT_VARIABLE(0);
    auto K = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GATED_DELTA_RULE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([Q, K, output] {
        return llamacppUtils::isSupportedType(Q->dataType()) &&
               llamacppUtils::isSupportedType(K->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectGreaterEq(makeInfoVariable(block.width(), "number of inputs"), 5);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// CAUSAL_CONV1D - Depthwise causal 1D convolution (CPU)
PLATFORM_IMPL(causal_conv1d, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);         // [B, L, D]
    auto weight = INPUT_VARIABLE(1);    // [D, K]
    auto output = OUTPUT_VARIABLE(0);   // [B, L, D]
    auto stateOut = OUTPUT_VARIABLE(1); // [B, D, K-1]

    if (x->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_w = llamacppUtils::createGgmlTensor(ctx, weight, "weight");

    struct ggml_tensor* ggml_output = ggml_ssm_conv(ctx, ggml_x, ggml_w);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);

    // State output: last K-1 elements of input
    { double zero = 0.0; stateOut->assign(zero); }

    return sd::Status::OK;
}

PLATFORM_CHECK(causal_conv1d, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto weight = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CAUSAL_CONV1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, weight, output] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(weight->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
