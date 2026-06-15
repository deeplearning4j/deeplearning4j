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
// Apple Accelerate framework - Matrix multiplication via vecLib BLAS
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

/**
 * Perform matrix multiplication using Accelerate's CBLAS
 * C = alpha * op(A) * op(B) + beta * C
 */
static void matmulAccelerate(const NDArray* a, const NDArray* b, NDArray* c,
                              bool transA, bool transB,
                              double alpha, double beta) {
    // Get dimensions
    // For matrix multiplication: C[M,N] = A[M,K] * B[K,N]
    const int M = transA ? a->sizeAt(1) : a->sizeAt(0);
    const int N = transB ? b->sizeAt(0) : b->sizeAt(1);
    const int K = transA ? a->sizeAt(0) : a->sizeAt(1);

    // Get leading dimensions (strides)
    const int lda = a->strideAt(0);
    const int ldb = b->strideAt(0);
    const int ldc = c->strideAt(0);

    // Determine transpose operations
    CBLAS_TRANSPOSE transAOp = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transBOp = transB ? CblasTrans : CblasNoTrans;

    if (a->dataType() == DataType::FLOAT32) {
        cblas_sgemm(CblasRowMajor, transAOp, transBOp,
                    M, N, K,
                    static_cast<float>(alpha),
                    a->bufferAsT<float>(), lda,
                    b->bufferAsT<float>(), ldb,
                    static_cast<float>(beta),
                    c->bufferAsT<float>(), ldc);
    } else if (a->dataType() == DataType::DOUBLE) {
        cblas_dgemm(CblasRowMajor, transAOp, transBOp,
                    M, N, K,
                    alpha,
                    a->bufferAsT<double>(), lda,
                    b->bufferAsT<double>(), ldb,
                    beta,
                    c->bufferAsT<double>(), ldc);
    }
}

PLATFORM_IMPL(matmul, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    // Get transpose flags from integer arguments
    // iArgs[0] = transA, iArgs[1] = transB, iArgs[2] = transC (unused)
    int transA = 0;
    int transB = 0;

    if (block.getIArguments()->size() > 0)
        transA = INT_ARG(0);
    if (block.getIArguments()->size() > 1)
        transB = INT_ARG(1);

    // Get alpha/beta from T arguments (default to 1.0 and 0.0)
    double alpha = 1.0;
    double beta = 0.0;

    if (block.getTArguments()->size() > 0)
        alpha = T_ARG(0);
    if (block.getTArguments()->size() > 1)
        beta = T_ARG(1);

    // Handle batch matmul for 3D+ tensors
    if (a->rankOf() > 2) {
        // For batched operations, we need to iterate over the batch dimensions
        // This is a simplified implementation - for optimal performance,
        // consider using vDSP batch operations
        auto aReshape = a->reshape(a->ordering(), {-1, a->sizeAt(-2), a->sizeAt(-1)});
        auto bReshape = b->reshape(b->ordering(), {-1, b->sizeAt(-2), b->sizeAt(-1)});
        auto cReshape = c->reshape(c->ordering(), {-1, c->sizeAt(-2), c->sizeAt(-1)});

        LongType batchSize = aReshape.sizeAt(0);

        for (LongType i = 0; i < batchSize; i++) {
            auto aSub = aReshape.subarray({NDIndex::point(i), NDIndex::all(), NDIndex::all()});
            auto bSub = bReshape.subarray({NDIndex::point(i), NDIndex::all(), NDIndex::all()});
            auto cSub = cReshape.subarray({NDIndex::point(i), NDIndex::all(), NDIndex::all()});

            matmulAccelerate(&aSub, &bSub, &cSub, transA != 0, transB != 0, alpha, beta);
        }
    } else {
        // Standard 2D matrix multiplication
        matmulAccelerate(a, b, c, transA != 0, transB != 0, alpha, beta);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(matmul, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements req("ACCELERATE MATMUL OP");

    // Check if Accelerate is enabled
    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);

    // Check data type support (float32 and float64 only for BLAS)
    req.expectTrue(a->dataType() == DataType::FLOAT32 || a->dataType() == DataType::DOUBLE,
                   "Only float32 and float64 are supported by Accelerate BLAS");

    // Data types must match
    req.expectTrue(a->dataType() == b->dataType(),
                   "Input arrays must have the same data type");

    // Arrays must be contiguous for optimal performance
    req.expectTrue(accelerateUtils::isContiguous(*a),
                   "Input A must be contiguous (ews=1)");
    req.expectTrue(accelerateUtils::isContiguous(*b),
                   "Input B must be contiguous (ews=1)");

    // Arrays must not be empty
    req.expectFalse(a->isEmpty(), "Input A must not be empty");
    req.expectFalse(b->isEmpty(), "Input B must not be empty");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
