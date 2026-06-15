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
 * Quantized Matrix Multiplication using GGML
 *
 * Supports various quantization formats:
 * - Q4_0, Q4_1: 4-bit quantization
 * - Q5_0, Q5_1: 5-bit quantization
 * - Q8_0, Q8_1: 8-bit quantization
 * - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K: K-quant variants
 */

enum QuantType {
    QUANT_Q4_0 = 0,
    QUANT_Q4_1 = 1,
    QUANT_Q5_0 = 2,
    QUANT_Q5_1 = 3,
    QUANT_Q8_0 = 4,
    QUANT_Q8_1 = 5,
    QUANT_Q2_K = 6,
    QUANT_Q3_K = 7,
    QUANT_Q4_K = 8,
    QUANT_Q5_K = 9,
    QUANT_Q6_K = 10
};

static ggml_type quantTypeToGgml(int quantType) {
    switch (quantType) {
        case QUANT_Q4_0: return GGML_TYPE_Q4_0;
        case QUANT_Q4_1: return GGML_TYPE_Q4_1;
        case QUANT_Q5_0: return GGML_TYPE_Q5_0;
        case QUANT_Q5_1: return GGML_TYPE_Q5_1;
        case QUANT_Q8_0: return GGML_TYPE_Q8_0;
        case QUANT_Q8_1: return GGML_TYPE_Q8_1;
        case QUANT_Q2_K: return GGML_TYPE_Q2_K;
        case QUANT_Q3_K: return GGML_TYPE_Q3_K;
        case QUANT_Q4_K: return GGML_TYPE_Q4_K;
        case QUANT_Q5_K: return GGML_TYPE_Q5_K;
        case QUANT_Q6_K: return GGML_TYPE_Q6_K;
        default: return GGML_TYPE_Q4_0;
    }
}

static void quantizedMatmulLlamaCpp(NDArray* x, NDArray* y, NDArray* z,
                                     const bool transX, const bool transY,
                                     int quantType) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);  // 128MB for quantized ops

    // Handle transposition
    std::vector<sd::LongType> permut;
    const auto xRank = x->rankOf();
    const auto yRank = y->rankOf();

    if ((transX && xRank > 1) || (transY && yRank > 1)) {
        const int rank = xRank >= yRank ? xRank : yRank;
        permut.resize(rank);
        std::iota(std::begin(permut), std::end(permut), 0);
        permut[rank - 2] = rank - 1;
        permut[rank - 1] = rank - 2;
    }

    NDArray* xT = (transX && xRank > 1) ? x->permute(permut, false, false) : x;
    NDArray* yT = (transY && yRank > 1) ? y->permute(permut, false, false) : y;

    // Create input tensor
    struct ggml_tensor* ggml_x = llamacppUtils::createGgmlTensor(ctx, xT, "x");

    // Create quantized weight tensor
    ggml_type qtype = quantTypeToGgml(quantType);
    const auto shape = yT->shapeOf();
    const auto rank = yT->rankOf();

    struct ggml_tensor* ggml_y;
    if (rank == 2) {
        ggml_y = ggml_new_tensor_2d(ctx, qtype, shape[1], shape[0]);
    } else if (rank == 3) {
        ggml_y = ggml_new_tensor_3d(ctx, qtype, shape[2], shape[1], shape[0]);
    } else {
        ggml_y = ggml_new_tensor_4d(ctx, qtype,
            rank >= 4 ? shape[3] : 1,
            rank >= 3 ? shape[2] : 1,
            shape[1], shape[0]);
    }

    // Quantize the weights
    const size_t srcSize = yT->lengthOf() * yT->sizeOfT();
    const size_t dstSize = ggml_nbytes(ggml_y);

    // Perform quantization
    ggml_quantize_chunk(qtype, (const float*)yT->buffer(), ggml_y->data,
                        0, yT->rows(), yT->columns(), nullptr);

    ggml_set_name(ggml_y, "y_quantized");

    // Perform quantized matmul
    struct ggml_tensor* ggml_z = ggml_mul_mat(ctx, ggml_y, ggml_x);
    ggml_set_name(ggml_z, "z");

    // Build and execute
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_z);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    // Copy result back
    llamacppUtils::copyGgmlToNDArray(ggml_z, z);

    // Cleanup
    if (xT != x) delete xT;
    if (yT != y) delete yT;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(quantized_matmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    sd::LongType iSize = (sd::LongType)block.getIArguments()->size();
    int transX = iSize > 0 ? INT_ARG(0) : 0;
    int transY = iSize > 1 ? INT_ARG(1) : 0;
    int quantType = iSize > 2 ? INT_ARG(2) : QUANT_Q4_0;  // Default to Q4_0

    quantizedMatmulLlamaCpp(x, y, z, transX, transY, quantType);

    return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(quantized_matmul, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP QUANTIZED_MATMUL OP");

    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);

    // Only float32 input supported for quantized matmul
    req.expectTrue(
        makeInfoVariable(
            [x] { return x->dataType() == DataType::FLOAT32; },
            TYPECHECK_MSG),
        NO_MSG);

    // Check rank constraints
    req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectLessEq(makeInfoVariable(y->rankOf(), RANK_MSG_INPUT1), 4);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
