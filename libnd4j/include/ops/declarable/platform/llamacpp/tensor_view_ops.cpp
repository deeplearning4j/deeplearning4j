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
// Tensor view and copy operations using GGML kernels:
// dup, cpy, cont, reshape, view, set
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
// DUP - Duplicate tensor
static void dupLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_dup(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(dup, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    dupLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(dup, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP DUP OP");
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
// CONT - Make tensor contiguous
static void contLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_cont(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(contiguous, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    contLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(contiguous, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CONTIGUOUS OP");
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
// RESHAPE - Reshape tensor
static void reshapeLlamaCpp(NDArray* input, NDArray* output, const std::vector<sd::LongType>& newShape) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output;
    if (newShape.size() == 1) {
        ggml_output = ggml_reshape_1d(ctx, ggml_input, newShape[0]);
    } else if (newShape.size() == 2) {
        ggml_output = ggml_reshape_2d(ctx, ggml_input, newShape[1], newShape[0]);
    } else if (newShape.size() == 3) {
        ggml_output = ggml_reshape_3d(ctx, ggml_input, newShape[2], newShape[1], newShape[0]);
    } else {
        ggml_output = ggml_reshape_4d(ctx, ggml_input,
            newShape.size() > 3 ? newShape[3] : 1,
            newShape.size() > 2 ? newShape[2] : 1,
            newShape[1], newShape[0]);
    }
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(reshape, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<sd::LongType> newShape;
    for (int i = 0; i < output->rankOf(); i++) {
        newShape.push_back(output->sizeAt(i));
    }

    reshapeLlamaCpp(input, output, newShape);
    return sd::Status::OK;
}

PLATFORM_CHECK(reshape, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP RESHAPE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// CPY - Copy tensor to different type/layout
static void cpyLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Create output tensor with target type
    const auto shape = output->shapeOf();
    const auto rank = output->rankOf();

    struct ggml_tensor* ggml_dst;
    if (rank == 1) {
        ggml_dst = ggml_new_tensor_1d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[0]);
    } else if (rank == 2) {
        ggml_dst = ggml_new_tensor_2d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[1], shape[0]);
    } else if (rank == 3) {
        ggml_dst = ggml_new_tensor_3d(ctx, llamacppUtils::toGgmlType(output->dataType()), shape[2], shape[1], shape[0]);
    } else {
        ggml_dst = ggml_new_tensor_4d(ctx, llamacppUtils::toGgmlType(output->dataType()),
            rank >= 4 ? shape[3] : 1, rank >= 3 ? shape[2] : 1, shape[1], shape[0]);
    }

    struct ggml_tensor* ggml_output = ggml_cpy(ctx, ggml_input, ggml_dst);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(copy, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    cpyLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(copy, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP COPY OP");
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
// SET - Set values in tensor
static void setLlamaCpp(NDArray* dst, NDArray* src, NDArray* output,
                        size_t offset, size_t nb1, size_t nb2, size_t nb3) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_dst = llamacppUtils::createGgmlTensor(ctx, dst, "dst");
    struct ggml_tensor* ggml_src = llamacppUtils::createGgmlTensor(ctx, src, "src");

    struct ggml_tensor* ggml_output = ggml_set(ctx, ggml_dst, ggml_src, nb1, nb2, nb3, offset);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(assign, ENGINE_CPU) {
    auto dst = INPUT_VARIABLE(0);
    auto src = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (dst->isEmpty() || src->isEmpty()) return sd::Status::OK;

    // Simple copy for now - more complex set with offsets would need additional args
    cpyLlamaCpp(src, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(assign, ENGINE_CPU) {
    auto dst = INPUT_VARIABLE(0);
    auto src = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ASSIGN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([dst, src, output] {
        return llamacppUtils::isSupportedType(dst->dataType()) &&
               llamacppUtils::isSupportedType(src->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(dst->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// FLATTEN - Flatten tensor to 1D
static void flattenLlamaCpp(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Reshape to 1D
    int64_t totalElements = 1;
    for (int i = 0; i < input->rankOf(); i++) {
        totalElements *= input->sizeAt(i);
    }

    struct ggml_tensor* ggml_output = ggml_reshape_1d(ctx, ggml_input, totalElements);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(flatten, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    flattenLlamaCpp(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(flatten, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP FLATTEN OP");
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
