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
// RWKV (Receptance Weighted Key Value) operations using GGML kernels:
// rwkv_wkv6, rwkv_wkv7 (RWKV v6 and v7 architecture support)
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
// RWKV_WKV6 - RWKV v6 Weighted Key-Value operation
// The core attention mechanism for RWKV-v6 architecture
static void rwkvWkv6LlamaCpp(NDArray* k, NDArray* v, NDArray* r, NDArray* tf, NDArray* td, NDArray* state, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, k, "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, v, "v");
    struct ggml_tensor* ggml_r = llamacppUtils::createGgmlTensor(ctx, r, "r");
    struct ggml_tensor* ggml_tf = llamacppUtils::createGgmlTensor(ctx, tf, "tf");
    struct ggml_tensor* ggml_td = llamacppUtils::createGgmlTensor(ctx, td, "td");
    struct ggml_tensor* ggml_state = llamacppUtils::createGgmlTensor(ctx, state, "state");

    struct ggml_tensor* ggml_output = ggml_rwkv_wkv6(ctx, ggml_k, ggml_v, ggml_r, ggml_tf, ggml_td, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(rwkv_wkv6, ENGINE_CPU) {
    auto k = INPUT_VARIABLE(0);      // Key
    auto v = INPUT_VARIABLE(1);      // Value
    auto r = INPUT_VARIABLE(2);      // Receptance
    auto tf = INPUT_VARIABLE(3);     // Time first
    auto td = INPUT_VARIABLE(4);     // Time decay
    auto state = INPUT_VARIABLE(5);  // State
    auto output = OUTPUT_VARIABLE(0);

    if (k->isEmpty() || v->isEmpty()) return sd::Status::OK;

    rwkvWkv6LlamaCpp(k, v, r, tf, td, state, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(rwkv_wkv6, ENGINE_CPU) {
    auto k = INPUT_VARIABLE(0);
    auto v = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP RWKV_WKV6 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([k, v, output] {
        return llamacppUtils::isSupportedType(k->dataType()) &&
               llamacppUtils::isSupportedType(v->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    // Expect 6 inputs: k, v, r, tf, td, state
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 6);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// RWKV_WKV7 - RWKV v7 Weighted Key-Value operation
// Enhanced attention mechanism for RWKV-v7 architecture with additional parameters
static void rwkvWkv7LlamaCpp(NDArray* r, NDArray* w, NDArray* k, NDArray* v, NDArray* a, NDArray* b, NDArray* state, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_r = llamacppUtils::createGgmlTensor(ctx, r, "r");
    struct ggml_tensor* ggml_w = llamacppUtils::createGgmlTensor(ctx, w, "w");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensor(ctx, k, "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensor(ctx, v, "v");
    struct ggml_tensor* ggml_a = llamacppUtils::createGgmlTensor(ctx, a, "a");
    struct ggml_tensor* ggml_b = llamacppUtils::createGgmlTensor(ctx, b, "b");
    struct ggml_tensor* ggml_state = llamacppUtils::createGgmlTensor(ctx, state, "state");

    struct ggml_tensor* ggml_output = ggml_rwkv_wkv7(ctx, ggml_r, ggml_w, ggml_k, ggml_v, ggml_a, ggml_b, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(rwkv_wkv7, ENGINE_CPU) {
    auto r = INPUT_VARIABLE(0);      // Receptance
    auto w = INPUT_VARIABLE(1);      // Weight
    auto k = INPUT_VARIABLE(2);      // Key
    auto v = INPUT_VARIABLE(3);      // Value
    auto a = INPUT_VARIABLE(4);      // A parameter
    auto b = INPUT_VARIABLE(5);      // B parameter
    auto state = INPUT_VARIABLE(6);  // State
    auto output = OUTPUT_VARIABLE(0);

    if (k->isEmpty() || v->isEmpty()) return sd::Status::OK;

    rwkvWkv7LlamaCpp(r, w, k, v, a, b, state, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(rwkv_wkv7, ENGINE_CPU) {
    auto k = INPUT_VARIABLE(2);
    auto v = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP RWKV_WKV7 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([k, v, output] {
        return llamacppUtils::isSupportedType(k->dataType()) &&
               llamacppUtils::isSupportedType(v->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    // Expect 7 inputs: r, w, k, v, a, b, state
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 7);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
