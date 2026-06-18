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
// Elementwise math operations using GGML CUDA kernels:
// abs, neg, sqr, sqrt, exp, log, sin, cos, tanh, ceil, floor, round, sign
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

// Helper macro for simple unary operations on CUDA
#define DEFINE_CUDA_UNARY_OP(OP_NAME, GGML_FUNC) \
static void OP_NAME##Cuda(NDArray* input, NDArray* output) { \
    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024); \
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input"); \
    struct ggml_tensor* ggml_output = GGML_FUNC(ctx, ggml_input); \
    ggml_set_name(ggml_output, "output"); \
    struct ggml_cgraph* graph = ggml_new_graph(ctx); \
    ggml_build_forward_expand(graph, ggml_output); \
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend()); \
    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend()); \
} \
\
PLATFORM_IMPL(OP_NAME, ENGINE_CUDA) { \
    auto input = INPUT_VARIABLE(0); \
    auto output = OUTPUT_VARIABLE(0); \
    if (input->isEmpty()) return sd::Status::OK; \
    OP_NAME##Cuda(input, output); \
    return sd::Status::OK; \
} \
\
PLATFORM_CHECK(OP_NAME, ENGINE_CUDA) { \
    auto input = INPUT_VARIABLE(0); \
    auto output = OUTPUT_VARIABLE(0); \
    Requirements req("LLAMACPP CUDA " #OP_NAME " OP"); \
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG); \
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend, "CUDA backend available"), NO_MSG); \
    req.expectTrue(makeInfoVariable([input, output] { \
        return llamacppUtils::isSupportedType(input->dataType()) && \
               llamacppUtils::isSupportedType(output->dataType()); \
    }, TYPECHECK_MSG), NO_MSG); \
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4); \
    req.logTheSuccess(); \
    return req; \
}

//////////////////////////////////////////////////////////////////////////
// Define all elementwise math operations

DEFINE_CUDA_UNARY_OP(abs, ggml_abs)
DEFINE_CUDA_UNARY_OP(neg, ggml_neg)
DEFINE_CUDA_UNARY_OP(square, ggml_sqr)
DEFINE_CUDA_UNARY_OP(sqrt, ggml_sqrt)
DEFINE_CUDA_UNARY_OP(exp, ggml_exp)
DEFINE_CUDA_UNARY_OP(log, ggml_log)
DEFINE_CUDA_UNARY_OP(sin, ggml_sin)
DEFINE_CUDA_UNARY_OP(cos, ggml_cos)
DEFINE_CUDA_UNARY_OP(tanh, ggml_tanh)
DEFINE_CUDA_UNARY_OP(ceil, ggml_ceil)
DEFINE_CUDA_UNARY_OP(floor, ggml_floor)
DEFINE_CUDA_UNARY_OP(round, ggml_round)
DEFINE_CUDA_UNARY_OP(sign, ggml_sgn)

#undef DEFINE_CUDA_UNARY_OP

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
