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
 * Rotary Position Embedding (RoPE) using GGML
 *
 * RoPE is a key component in LLaMA and many modern transformer models.
 * It encodes positional information by rotating the query and key vectors.
 */
static void ropeLlamaCpp(NDArray* input, NDArray* output,
                         int nPast, int nDims, int mode,
                         int nCtx, int nOrigCtx,
                         float freqBase, float freqScale,
                         float extFactor, float attnFactor,
                         float betaFast, float betaSlow) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Apply RoPE
    struct ggml_tensor* ggml_output = ggml_rope_ext(
        ctx, ggml_input,
        nullptr,  // position tensor (optional)
        nullptr,  // frequency factors (optional)
        nDims, mode, nCtx, nOrigCtx,
        freqBase, freqScale,
        extFactor, attnFactor,
        betaFast, betaSlow
    );

    ggml_set_name(ggml_output, "output");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(rope, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // Integer arguments
    int nPast = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int nDims = block.getIArguments()->size() > 1 ? INT_ARG(1) : 64;
    int mode = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;
    int nCtx = block.getIArguments()->size() > 3 ? INT_ARG(3) : 2048;
    int nOrigCtx = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;

    // Float arguments
    float freqBase = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0f;
    float freqScale = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;
    float extFactor = block.getTArguments()->size() > 2 ? T_ARG(2) : 0.0f;
    float attnFactor = block.getTArguments()->size() > 3 ? T_ARG(3) : 1.0f;
    float betaFast = block.getTArguments()->size() > 4 ? T_ARG(4) : 32.0f;
    float betaSlow = block.getTArguments()->size() > 5 ? T_ARG(5) : 1.0f;

    ropeLlamaCpp(input, output, nPast, nDims, mode, nCtx, nOrigCtx,
                 freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(rope, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ROPE OP");

    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    req.expectTrue(
        makeInfoVariable(
            [input, output] {
                return llamacppUtils::isSupportedType(input->dataType()) &&
                       llamacppUtils::isSupportedType(output->dataType());
            },
            TYPECHECK_MSG),
        NO_MSG);

    // RoPE typically requires 3D or 4D tensors
    req.expectGreaterEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
