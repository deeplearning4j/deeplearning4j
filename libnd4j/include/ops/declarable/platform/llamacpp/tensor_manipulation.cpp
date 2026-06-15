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
// Tensor manipulation operations using GGML kernels:
// concat, pad, repeat, upscale, get_rows, diag, permute, transpose, reshape
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
// CONCAT - Concatenate tensors along an axis
static void concatLlamaCpp(std::vector<NDArray*>& inputs, NDArray* output, int axis) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    if (inputs.size() < 2) {
        // Single input, just copy
        if (inputs.size() == 1) {
            output->assign(inputs[0]);
        }
        return;
    }

    // Create GGML tensors for all inputs
    struct ggml_tensor* result = llamacppUtils::createGgmlTensor(ctx, inputs[0], "input_0");

    for (size_t i = 1; i < inputs.size(); i++) {
        std::string name = "input_" + std::to_string(i);
        struct ggml_tensor* next = llamacppUtils::createGgmlTensor(ctx, inputs[i], name.c_str());
        result = ggml_concat(ctx, result, next, axis);
    }

    ggml_set_name(result, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(result, output);
}

PLATFORM_IMPL(concat, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    // Get axis - typically last argument or default to 0
    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    // Handle negative axis
    if (axis < 0) {
        axis += output->rankOf();
    }

    // Collect all inputs
    std::vector<NDArray*> inputs;
    for (int i = 0; i < block.width(); i++) {
        auto input = INPUT_VARIABLE(i);
        if (!input->isEmpty()) {
            inputs.push_back(input);
        }
    }

    if (inputs.empty()) return sd::Status::OK;

    concatLlamaCpp(inputs, output, axis);
    return sd::Status::OK;
}

PLATFORM_CHECK(concat, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CONCAT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// PAD - Pad tensor with constant value
static void padLlamaCpp(NDArray* input, NDArray* output,
                        int p0, int p1, int p2, int p3,
                        int p4, int p5, int p6, int p7) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_pad(ctx, ggml_input, p0, p1, p2, p3);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(pad, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // Get padding values
    int p0 = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int p1 = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
    int p2 = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;
    int p3 = block.getIArguments()->size() > 3 ? INT_ARG(3) : 0;
    int p4 = block.getIArguments()->size() > 4 ? INT_ARG(4) : 0;
    int p5 = block.getIArguments()->size() > 5 ? INT_ARG(5) : 0;
    int p6 = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;
    int p7 = block.getIArguments()->size() > 7 ? INT_ARG(7) : 0;

    padLlamaCpp(input, output, p0, p1, p2, p3, p4, p5, p6, p7);
    return sd::Status::OK;
}

PLATFORM_CHECK(pad, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP PAD OP");
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
// REPEAT - Repeat tensor along dimensions
static void repeatLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Create output tensor shape for repeat
    struct ggml_tensor* ggml_shape = llamacppUtils::createGgmlTensor(ctx, output, "shape");
    struct ggml_tensor* ggml_output = ggml_repeat(ctx, ggml_input, ggml_shape);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(repeat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    repeatLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(repeat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP REPEAT OP");
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
// UPSCALE - Upscale tensor (nearest neighbor interpolation)
static void upscaleLlamaCpp(NDArray* input, NDArray* output, int scaleFactor) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_upscale(ctx, ggml_input, scaleFactor);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(upsampling2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int scaleFactor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 2;

    upscaleLlamaCpp(input, output, scaleFactor);
    return sd::Status::OK;
}

PLATFORM_CHECK(upsampling2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP UPSAMPLING2D OP");
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
// GET_ROWS - Gather rows from tensor using indices
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

PLATFORM_IMPL(gather, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    getRowsLlamaCpp(input, indices, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(gather, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP GATHER OP");
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
// DIAG - Create diagonal matrix or extract diagonal
static void diagLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_diag(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(diag, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    diagLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(diag, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP DIAG OP");
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
// TRANSPOSE - Transpose tensor
static void transposeLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_transpose(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(transpose, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // DSP view path: output shares input's buffer with transposed strides.
    if (input->dataBuffer() == output->dataBuffer()) return sd::Status::OK;

    transposeLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(transpose, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP TRANSPOSE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    // GGML transpose works on 2D tensors
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// PERMUTE - Permute tensor dimensions
static void permuteLlamaCpp(NDArray* input, NDArray* output, int axis0, int axis1, int axis2, int axis3) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_output = ggml_permute(ctx, ggml_input, axis0, axis1, axis2, axis3);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(permute, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // DSP view path: output shares input's buffer with permuted strides.
    if (input->dataBuffer() == output->dataBuffer()) return sd::Status::OK;

    // Get permutation axes
    int axis0 = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int axis1 = block.getIArguments()->size() > 1 ? INT_ARG(1) : 1;
    int axis2 = block.getIArguments()->size() > 2 ? INT_ARG(2) : 2;
    int axis3 = block.getIArguments()->size() > 3 ? INT_ARG(3) : 3;

    permuteLlamaCpp(input, output, axis0, axis1, axis2, axis3);
    return sd::Status::OK;
}

PLATFORM_CHECK(permute, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP PERMUTE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
