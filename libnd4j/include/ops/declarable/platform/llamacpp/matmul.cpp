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
 * Matrix multiplication using GGML kernels from llama.cpp
 *
 * This implementation uses GGML's highly optimized matrix multiplication
 * which includes:
 * - AVX/AVX2/AVX512 vectorization on x86
 * - NEON optimization on ARM
 * - Support for various quantized formats
 */
static void matmulLlamaCpp(NDArray* x, NDArray* y, NDArray* z,
                           const bool transX, const bool transY,
                           float alpha = 1.f, float beta = 0.f) {
    // Get GGML context
    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);  // 64MB workspace

    const auto xRank = x->rankOf();
    const auto yRank = y->rankOf();

    // Handle transposition by permuting if needed
    std::vector<sd::LongType> permut;
    if ((transX && xRank > 1) || (transY && yRank > 1)) {
        const int rank = xRank >= yRank ? xRank : yRank;
        permut.resize(rank);
        std::iota(std::begin(permut), std::end(permut), 0);
        permut[rank - 2] = rank - 1;
        permut[rank - 1] = rank - 2;
    }

    NDArray* xT = (transX && xRank > 1) ? x->permute(permut, false, false) : x;
    NDArray* yT = (transY && yRank > 1) ? y->permute(permut, false, false) : y;

    // Create GGML tensors
    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, xT, "x");
    struct ggml_tensor* ggml_y = llamacppUtils::createGgmlTensor(ctx, yT, "y");

    // Perform matrix multiplication
    struct ggml_tensor* ggml_z = ggml_mul_mat(ctx, ggml_x, ggml_y);
    ggml_set_name(ggml_z, "z");

    // Handle alpha scaling
    if (alpha != 1.0f) {
        struct ggml_tensor* alpha_tensor = ggml_new_f32(ctx, alpha);
        ggml_z = ggml_scale(ctx, ggml_z, alpha_tensor);
    }

    // Handle beta (add existing z values)
    if (beta != 0.0f && z->buffer() != nullptr) {
        struct ggml_tensor* ggml_z_orig = llamacppUtils::createGgmlTensor(ctx, z, "z_orig");
        struct ggml_tensor* beta_tensor = ggml_new_f32(ctx, beta);
        struct ggml_tensor* scaled_z_orig = ggml_scale(ctx, ggml_z_orig, beta_tensor);
        ggml_z = ggml_add(ctx, ggml_z, scaled_z_orig);
    }

    // Build and execute computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);

    llamacppUtils::executeGgmlGraph(ctx, graph);

    // Copy result back to output NDArray
    llamacppUtils::copyGgmlToNDArray(ggml_z, z);

    // Cleanup temporary arrays
    if (xT != x) delete xT;
    if (yT != y) delete yT;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(matmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    sd::LongType iSize = (sd::LongType)block.getIArguments()->size();
    int transX = iSize > 0 ? INT_ARG(0) : 0;
    int transY = iSize > 1 ? INT_ARG(1) : 0;
    const int transZ = iSize > 2 ? INT_ARG(2) : 0;

    // Optional alpha and beta
    iSize = (sd::LongType)block.getTArguments()->size();
    float alpha = iSize > 0 ? T_ARG(0) : 1.0f;
    float beta = iSize > 1 ? T_ARG(1) : 0.0f;

    const sd::LongType xRank = x->rankOf();
    const sd::LongType yRank = y->rankOf();
    const sd::LongType zRank = z->rankOf();

    if (transZ) {
        x = INPUT_VARIABLE(1);
        y = INPUT_VARIABLE(0);
        bool temp = transX;
        transX = !transY;
        transY = !temp;
    }

    // Input validation
    REQUIRE_TRUE(xRank > 0 && yRank > 0, 0,
                 "MATMUL LLAMACPP OP: input arrays must have rank bigger than 0, but got: "
                 "x rank = %i, y rank = %i !",
                 xRank, yRank);

    if (xRank == 1 && yRank == 1) {
        REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0,
                     "MATMUL LLAMACPP OP: vectors must have same length, got x = %i, y = %i !",
                     x->lengthOf(), y->lengthOf());
    }

    matmulLlamaCpp(x, y, z, transX, transY, alpha, beta);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(matmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const auto xType = x->dataType();
    const auto yType = y->dataType();
    const auto zType = z->dataType();

    Requirements req("LLAMACPP MATMUL OP");

    // Check if LLAMACPP backend is enabled
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    // Check supported data types
    req.expectTrue(
        makeInfoVariable(
            [xType, yType, zType] {
                return llamacppUtils::isSupportedType(xType) &&
                       llamacppUtils::isSupportedType(yType) &&
                       llamacppUtils::isSupportedType(zType);
            },
            TYPECHECK_MSG),
        NO_MSG);

    // Check rank constraints (GGML supports up to 4D)
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
