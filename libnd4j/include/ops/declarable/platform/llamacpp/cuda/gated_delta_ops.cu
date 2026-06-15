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
// CUDA implementations of Gated Delta Network operations using GGML kernels
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
// GATED_DELTA_RULE - Gated Delta Network recurrent state update (CUDA)
PLATFORM_IMPL(gated_delta_rule, ENGINE_CUDA) {
    auto Q = INPUT_VARIABLE(0);
    auto K = INPUT_VARIABLE(1);
    auto V = INPUT_VARIABLE(2);
    auto beta = INPUT_VARIABLE(3);
    auto gate = INPUT_VARIABLE(4);
    auto output = OUTPUT_VARIABLE(0);
    auto stateOut = OUTPUT_VARIABLE(1);

    if (Q->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_q = llamacppUtils::createGgmlTensorCuda(ctx, Q, ctx.getBackend(), "q");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, K, ctx.getBackend(), "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, V, ctx.getBackend(), "v");
    struct ggml_tensor* ggml_beta = llamacppUtils::createGgmlTensorCuda(ctx, beta, ctx.getBackend(), "beta");
    struct ggml_tensor* ggml_gate = llamacppUtils::createGgmlTensorCuda(ctx, gate, ctx.getBackend(), "gate");

    NDArray* stateIn = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
    struct ggml_tensor* ggml_state = stateIn != nullptr ?
        llamacppUtils::createGgmlTensorCuda(ctx, stateIn, ctx.getBackend(), "state") : nullptr;

    struct ggml_tensor* ggml_output = ggml_gated_delta_net(ctx, ggml_q, ggml_k, ggml_v,
                                                            ggml_beta, ggml_gate, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    { double zero = 0.0; stateOut->assign(zero); }

    return sd::Status::OK;
}

PLATFORM_CHECK(gated_delta_rule, ENGINE_CUDA) {
    auto Q = INPUT_VARIABLE(0);
    auto K = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA GATED_DELTA_RULE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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
// CAUSAL_CONV1D - Depthwise causal 1D convolution (CUDA)
PLATFORM_IMPL(causal_conv1d, ENGINE_CUDA) {
    auto x = INPUT_VARIABLE(0);
    auto weight = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);
    auto stateOut = OUTPUT_VARIABLE(1);

    if (x->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensorCuda(ctx, x, ctx.getBackend(), "x");
    struct ggml_tensor* ggml_w = llamacppUtils::createGgmlTensorCuda(ctx, weight, ctx.getBackend(), "weight");

    struct ggml_tensor* ggml_output = ggml_ssm_conv(ctx, ggml_x, ggml_w);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    { double zero = 0.0; stateOut->assign(zero); }

    return sd::Status::OK;
}

PLATFORM_CHECK(causal_conv1d, ENGINE_CUDA) {
    auto x = INPUT_VARIABLE(0);
    auto weight = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA CAUSAL_CONV1D OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
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

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
