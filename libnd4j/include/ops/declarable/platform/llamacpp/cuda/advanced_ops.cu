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
// Advanced operations using GGML CUDA kernels:
// argsort, argmax, top_k, cumsum, clip_by_value, range, mul_scalar,
// reduce_mean, count_equal, diag_mask_inf
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

#include <ggml-cuda.h>
#include <ggml-backend.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// ARGSORT - Sort indices on CUDA
static void argsortCuda(NDArray* input, NDArray* output, bool descending) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    enum ggml_sort_order order = descending ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC;
    struct ggml_tensor* ggml_output = ggml_argsort(ctx, ggml_input, order);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(argsort, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    bool descending = block.getIArguments()->size() > 0 ? INT_ARG(0) != 0 : false;
    argsortCuda(input, output, descending);
    return sd::Status::OK;
}

PLATFORM_CHECK(argsort, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ARGSORT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ARGMAX - Find index of maximum on CUDA
static void argmaxCuda(NDArray* input, NDArray* output, int axis) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_argmax(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(argmax, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : -1;
    argmaxCuda(input, output, axis);
    return sd::Status::OK;
}

PLATFORM_CHECK(argmax, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA ARGMAX OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// TOP_K - Find top K elements on CUDA
static void topKCuda(NDArray* input, NDArray* values, NDArray* indices, int k) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_top_k(ctx, ggml_input, k);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, values, ctx.getBackend());
}

PLATFORM_IMPL(top_k, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto values = OUTPUT_VARIABLE(0);
    NDArray* indices = block.width() > 1 ? OUTPUT_VARIABLE(1) : nullptr;

    if (input->isEmpty()) return sd::Status::OK;

    int k = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    topKCuda(input, values, indices, k);
    return sd::Status::OK;
}

PLATFORM_CHECK(top_k, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA TOP_K OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// CUMSUM - Cumulative sum on CUDA
static void cumsumCuda(NDArray* input, NDArray* output, int axis, bool exclusive, bool reverse) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_sum(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(cumsum, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    bool exclusive = block.getIArguments()->size() > 1 ? INT_ARG(1) != 0 : false;
    bool reverse = block.getIArguments()->size() > 2 ? INT_ARG(2) != 0 : false;

    cumsumCuda(input, output, axis, exclusive, reverse);
    return sd::Status::OK;
}

PLATFORM_CHECK(cumsum, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA CUMSUM OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// CLIP_BY_VALUE - Clamp values on CUDA
static void clipByValueCuda(NDArray* input, NDArray* output, float minVal, float maxVal) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_clamp(ctx, ggml_input, minVal, maxVal);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(clip_by_value, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float minVal = block.getTArguments()->size() > 0 ? T_ARG(0) : -std::numeric_limits<float>::infinity();
    float maxVal = block.getTArguments()->size() > 1 ? T_ARG(1) : std::numeric_limits<float>::infinity();

    clipByValueCuda(input, output, minVal, maxVal);
    return sd::Status::OK;
}

PLATFORM_CHECK(clip_by_value, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA CLIP_BY_VALUE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// RANGE - Generate range on CUDA
static void rangeCuda(NDArray* output, float start, float stop, float step) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    int n = static_cast<int>((stop - start) / step);
    struct ggml_tensor* ggml_output = ggml_arange(ctx, start, stop, step);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(range, ENGINE_CUDA) {
    auto output = OUTPUT_VARIABLE(0);

    float start = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    float stop = block.getTArguments()->size() > 1 ? T_ARG(1) : 1.0f;
    float step = block.getTArguments()->size() > 2 ? T_ARG(2) : 1.0f;

    rangeCuda(output, start, stop, step);
    return sd::Status::OK;
}

PLATFORM_CHECK(range, ENGINE_CUDA) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA RANGE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MUL_SCALAR - Scalar multiplication on CUDA
static void mulScalarCuda(NDArray* input, NDArray* output, float scalar) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_scale(ctx, ggml_input, scalar);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(mul_scalar, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float scalar = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;
    mulScalarCuda(input, output, scalar);
    return sd::Status::OK;
}

PLATFORM_CHECK(mul_scalar, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA MUL_SCALAR OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// REDUCE_MEAN - Mean reduction on CUDA
static void reduceMeanCuda(NDArray* input, NDArray* output) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_mean(ctx, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(reduce_mean, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    reduceMeanCuda(input, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_mean, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA REDUCE_MEAN OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// COUNT_EQUAL - Count equal elements on CUDA
static void countEqualCuda(NDArray* input, NDArray* output, float value) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_count_equal(ctx, ggml_input, ggml_input);  // Compare with self for now
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(count_equal, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float value = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0f;
    countEqualCuda(input, output, value);
    return sd::Status::OK;
}

PLATFORM_CHECK(count_equal, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA COUNT_EQUAL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DIAG_MASK_INF - Diagonal mask with infinity on CUDA
static void diagMaskInfCuda(NDArray* input, NDArray* output, int nPast) {
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_diag_mask_inf(ctx, ggml_input, nPast);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
}

PLATFORM_IMPL(diag_mask_inf, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int nPast = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    diagMaskInfCuda(input, output, nPast);
    return sd::Status::OK;
}

PLATFORM_CHECK(diag_mask_inf, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA DIAG_MASK_INF OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 2);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
