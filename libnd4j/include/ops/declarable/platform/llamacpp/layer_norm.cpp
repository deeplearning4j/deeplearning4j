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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "llamacppUtils.h"

#if HAVE_LLAMACPP

namespace sd {
namespace ops {
namespace platforms {

/**
 * Layer Normalization using GGML
 *
 * LayerNorm(x) = (x - mean) / sqrt(variance + eps) * gamma + beta
 */
static void layerNormLlamaCpp(NDArray* input, NDArray* gamma, NDArray* beta,
                               NDArray* output, float eps) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Apply layer norm
    struct ggml_tensor* ggml_output = ggml_norm(ctx, ggml_input, eps);

    // Apply gamma (scale) if provided
    if (gamma != nullptr && !gamma->isEmpty()) {
        struct ggml_tensor* ggml_gamma = llamacppUtils::createGgmlTensor(ctx, gamma, "gamma");
        ggml_output = ggml_mul(ctx, ggml_output, ggml_gamma);
    }

    // Apply beta (bias) if provided
    if (beta != nullptr && !beta->isEmpty()) {
        struct ggml_tensor* ggml_beta = llamacppUtils::createGgmlTensor(ctx, beta, "beta");
        ggml_output = ggml_add(ctx, ggml_output, ggml_beta);
    }

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    NDArray* gamma = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
    NDArray* beta = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

    if (input->isEmpty()) return sd::Status::OK;

    float eps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    layerNormLlamaCpp(input, gamma, beta, output, eps);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP LAYER_NORM OP");

    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    req.expectTrue(
        makeInfoVariable(
            [input, output] {
                return llamacppUtils::isSupportedType(input->dataType()) &&
                       llamacppUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
