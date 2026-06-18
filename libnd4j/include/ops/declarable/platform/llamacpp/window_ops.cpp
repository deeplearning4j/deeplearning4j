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
// Window operations using GGML kernels:
// win_part, win_unpart (window partition and unpartition for vision transformers)
// geglu (Gated GELU activation)
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
// WIN_PART - Window Partition
// Partitions input tensor into non-overlapping windows (used in Swin Transformer)
static void winPartLlamaCpp(NDArray* input, NDArray* output, int windowSize) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_win_part(ctx, ggml_input, windowSize);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(win_part, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 7;

    winPartLlamaCpp(input, output, windowSize);
    return sd::Status::OK;
}

PLATFORM_CHECK(win_part, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP WIN_PART OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);  // NHWC format
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// WIN_UNPART - Window Unpartition
// Reverses window partition operation (used in Swin Transformer)
static void winUnpartLlamaCpp(NDArray* input, NDArray* output, int windowSize, int h, int w) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_win_unpart(ctx, ggml_input, windowSize, h, w);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(win_unpart, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int windowSize = block.getIArguments()->size() > 0 ? INT_ARG(0) : 7;
    int h = block.getIArguments()->size() > 1 ? INT_ARG(1) : output->sizeAt(1);
    int w = block.getIArguments()->size() > 2 ? INT_ARG(2) : output->sizeAt(2);

    winUnpartLlamaCpp(input, output, windowSize, h, w);
    return sd::Status::OK;
}

PLATFORM_CHECK(win_unpart, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP WIN_UNPART OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT0), 4);  // NHWC format
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GEGLU - Gated GELU activation
// Splits input in half, applies GELU to first half, multiplies with second half
static void gegluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_geglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(geglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    gegluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(geglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GEGLU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    // Input last dimension must be even (splits in half)
    req.expectTrue(makeInfoVariable(input->sizeAt(-1) % 2 == 0, "last dimension must be even"), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// REGLU - Gated ReLU activation (variant of GLU)
static void regluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_reglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(reglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    regluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(reglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP REGLU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(makeInfoVariable(input->sizeAt(-1) % 2 == 0, "last dimension must be even"), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SWIGLU - Gated SiLU/Swish activation
static void swigluLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_swiglu(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(swiglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    swigluLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(swiglu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SWIGLU OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(makeInfoVariable(input->sizeAt(-1) % 2 == 0, "last dimension must be even"), NO_MSG);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
