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
// Binary operations using GGML kernels:
// add, sub, mul, div
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
// ADD - Element-wise addition
static void addLlamaCpp(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, y, "y");

    struct ggml_tensor* ggml_z = ggml_add(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_z, z);
}

PLATFORM_IMPL(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    addLlamaCpp(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP ADD OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SUB - Element-wise subtraction
static void subLlamaCpp(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, y, "y");

    struct ggml_tensor* ggml_z = ggml_sub(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_z, z);
}

PLATFORM_IMPL(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    subLlamaCpp(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SUBTRACT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// MUL - Element-wise multiplication
static void mulLlamaCpp(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, y, "y");

    struct ggml_tensor* ggml_z = ggml_mul(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_z, z);
}

PLATFORM_IMPL(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    mulLlamaCpp(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP MULTIPLY OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DIV - Element-wise division
static void divLlamaCpp(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, y, "y");

    struct ggml_tensor* ggml_z = ggml_div(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_z, z);
}

PLATFORM_IMPL(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    divLlamaCpp(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP DIVIDE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// OUT_PROD - Outer product
static void outerProdLlamaCpp(NDArray* x, NDArray* y, NDArray* z) {
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, x, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, y, "y");

    struct ggml_tensor* ggml_z = ggml_out_prod(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_z, z);
}

PLATFORM_IMPL(tensormmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    outerProdLlamaCpp(x, y, z);
    return sd::Status::OK;
}

PLATFORM_CHECK(tensormmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP TENSORMMUL OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([x, y, z] {
        return llamacppUtils::isSupportedType(x->dataType()) &&
               llamacppUtils::isSupportedType(y->dataType()) &&
               llamacppUtils::isSupportedType(z->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
