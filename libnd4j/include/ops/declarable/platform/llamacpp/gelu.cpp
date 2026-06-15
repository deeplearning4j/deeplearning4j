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
 * GELU (Gaussian Error Linear Unit) activation using GGML
 *
 * GELU(x) = x * Phi(x) where Phi is the CDF of standard normal distribution
 * Used in BERT, GPT, and many transformer architectures.
 */
static void geluLlamaCpp(NDArray* input, NDArray* output, bool quick = false) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Apply GELU (quick variant uses faster approximation)
    struct ggml_tensor* ggml_output = quick ?
        ggml_gelu_quick(ctx, ggml_input) :
        ggml_gelu(ctx, ggml_input);

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(gelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // Optional: use quick GELU approximation
    bool quick = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : false;

    geluLlamaCpp(input, output, quick);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(gelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GELU OP");

    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    req.expectTrue(
        makeInfoVariable(
            [input, output] {
                return llamacppUtils::isSupportedType(input->dataType()) &&
                       llamacppUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
