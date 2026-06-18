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
 * Softmax using GGML kernels from llama.cpp
 */
static void softmaxLlamaCpp(NDArray* input, NDArray* output, int dimension) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);  // 32MB workspace

    // Create GGML tensor
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Apply softmax
    struct ggml_tensor* ggml_output = ggml_soft_max(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    // Build and execute computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);

    llamacppUtils::executeGgmlGraph(ctx, graph);

    // Copy result back
    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    const int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : input->rankOf() - 1;

    softmaxLlamaCpp(input, output, dimension);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SOFTMAX OP");

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
