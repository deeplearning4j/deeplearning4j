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
// State Space Model (SSM) operations using GGML kernels:
// ssm_conv, ssm_scan (Mamba architecture support)
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
// SSM_CONV - State Space Model Convolution
// Used in Mamba architecture for efficient sequence modeling
static void ssmConvLlamaCpp(NDArray* input, NDArray* conv, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_conv = llamacppUtils::createGgmlTensor(ctx, conv, "conv");

    struct ggml_tensor* ggml_output = ggml_ssm_conv(ctx, ggml_input, ggml_conv);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(ssm_conv, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto conv = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    ssmConvLlamaCpp(input, conv, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(ssm_conv, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto conv = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SSM_CONV OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, conv, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(conv->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SSM_SCAN - State Space Model Scan (selective scan operation)
// Core operation for Mamba sequence modeling
static void ssmScanLlamaCpp(NDArray* s, NDArray* x, NDArray* dt, NDArray* A, NDArray* B, NDArray* C, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_s = llamacppUtils::createGgmlTensor(ctx, s, "s");
    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_dt = llamacppUtils::createGgmlTensor(ctx, dt, "dt");
    struct ggml_tensor* ggml_A = llamacppUtils::createGgmlTensor(ctx, A, "A");
    struct ggml_tensor* ggml_B = llamacppUtils::createGgmlTensor(ctx, B, "B");
    struct ggml_tensor* ggml_C = llamacppUtils::createGgmlTensor(ctx, C, "C");

    struct ggml_tensor* ggml_output = ggml_ssm_scan(ctx, ggml_s, ggml_x, ggml_dt, ggml_A, ggml_B, ggml_C);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(ssm_scan, ENGINE_CPU) {
    auto s = INPUT_VARIABLE(0);      // State
    auto x = INPUT_VARIABLE(1);      // Input
    auto dt = INPUT_VARIABLE(2);     // Delta time
    auto A = INPUT_VARIABLE(3);      // State transition matrix
    auto B = INPUT_VARIABLE(4);      // Input matrix
    auto C = INPUT_VARIABLE(5);      // Output matrix
    auto output = OUTPUT_VARIABLE(0);

    if (s->isEmpty() || x->isEmpty()) return sd::Status::OK;

    ssmScanLlamaCpp(s, x, dt, A, B, C, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(ssm_scan, ENGINE_CPU) {
    auto s = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SSM_SCAN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([s, x, output] {
        return llamacppUtils::isSupportedType(s->dataType()) &&
               llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    // Expect 6 inputs: s, x, dt, A, B, C
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 6);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
