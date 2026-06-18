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
// CUDA implementations of RWKV (Receptance Weighted Key Value) operations using GGML kernels
// Supports RWKV v6 and v7 architectures
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// RWKV_WKV6 - RWKV v6 Weighted Key-Value (CUDA)
PLATFORM_IMPL(rwkv_wkv6, ENGINE_CUDA) {
    auto k = INPUT_VARIABLE(0);      // Key
    auto v = INPUT_VARIABLE(1);      // Value
    auto r = INPUT_VARIABLE(2);      // Receptance
    auto tf = INPUT_VARIABLE(3);     // Time first
    auto td = INPUT_VARIABLE(4);     // Time decay
    auto state = INPUT_VARIABLE(5);  // State
    auto output = OUTPUT_VARIABLE(0);

    if (k->isEmpty() || v->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, k, ctx.getBackend(), "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, v, ctx.getBackend(), "v");
    struct ggml_tensor* ggml_r = llamacppUtils::createGgmlTensorCuda(ctx, r, ctx.getBackend(), "r");
    struct ggml_tensor* ggml_tf = llamacppUtils::createGgmlTensorCuda(ctx, tf, ctx.getBackend(), "tf");
    struct ggml_tensor* ggml_td = llamacppUtils::createGgmlTensorCuda(ctx, td, ctx.getBackend(), "td");
    struct ggml_tensor* ggml_state = llamacppUtils::createGgmlTensorCuda(ctx, state, ctx.getBackend(), "state");

    struct ggml_tensor* ggml_output = ggml_rwkv_wkv6(ctx, ggml_k, ggml_v, ggml_r, ggml_tf, ggml_td, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(rwkv_wkv6, ENGINE_CUDA) {
    auto k = INPUT_VARIABLE(0);
    auto v = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA RWKV_WKV6 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([k, v, output] {
        return llamacppUtils::isSupportedType(k->dataType()) &&
               llamacppUtils::isSupportedType(v->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 6);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// RWKV_WKV7 - RWKV v7 Weighted Key-Value (CUDA)
PLATFORM_IMPL(rwkv_wkv7, ENGINE_CUDA) {
    auto r = INPUT_VARIABLE(0);      // Receptance
    auto w = INPUT_VARIABLE(1);      // Weight
    auto k = INPUT_VARIABLE(2);      // Key
    auto v = INPUT_VARIABLE(3);      // Value
    auto a = INPUT_VARIABLE(4);      // A parameter
    auto b = INPUT_VARIABLE(5);      // B parameter
    auto state = INPUT_VARIABLE(6);  // State
    auto output = OUTPUT_VARIABLE(0);

    if (k->isEmpty() || v->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_r = llamacppUtils::createGgmlTensorCuda(ctx, r, ctx.getBackend(), "r");
    struct ggml_tensor* ggml_w = llamacppUtils::createGgmlTensorCuda(ctx, w, ctx.getBackend(), "w");
    struct ggml_tensor* ggml_k = llamacppUtils::createGgmlTensorCuda(ctx, k, ctx.getBackend(), "k");
    struct ggml_tensor* ggml_v = llamacppUtils::createGgmlTensorCuda(ctx, v, ctx.getBackend(), "v");
    struct ggml_tensor* ggml_a = llamacppUtils::createGgmlTensorCuda(ctx, a, ctx.getBackend(), "a");
    struct ggml_tensor* ggml_b = llamacppUtils::createGgmlTensorCuda(ctx, b, ctx.getBackend(), "b");
    struct ggml_tensor* ggml_state = llamacppUtils::createGgmlTensorCuda(ctx, state, ctx.getBackend(), "state");

    struct ggml_tensor* ggml_output = ggml_rwkv_wkv7(ctx, ggml_r, ggml_w, ggml_k, ggml_v, ggml_a, ggml_b, ggml_state);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(rwkv_wkv7, ENGINE_CUDA) {
    auto k = INPUT_VARIABLE(2);
    auto v = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA RWKV_WKV7 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([k, v, output] {
        return llamacppUtils::isSupportedType(k->dataType()) &&
               llamacppUtils::isSupportedType(v->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectEq(makeInfoVariable(block.width(), "number of inputs"), 7);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
